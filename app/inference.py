from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from app.common import ROOT,read_json,sha256
from app.geometry import validate_box
from app.types import CLASSES,Detection


class Detector:
    def __init__(self,config):
        self.config=config;self.models=[];self.hashes=[];self.classes=set();self.error=None;self.policies=[]
        self.device="unavailable";self.provider="none";self.actual_device=None
        self.ready=False;self.degraded=True

    def load(self):
        os.environ.setdefault("YOLO_CONFIG_DIR",str(ROOT/".cache/ultralytics"))
        import torch
        from ultralytics import YOLO
        # Small CUDA inference batches benefit from avoiding CPU thread-pool overhead.
        torch.set_num_threads(max(1,int(self.config.get('inference_cpu_threads',1))))
        registry=read_json(ROOT/"models/model_registry.json",{"models":[]})
        chosen=[m for m in registry["models"] if m.get("enabled") and m.get("evaluation_status") in {"development_evaluated","accepted"}]
        if not chosen: self.error="no_newly_trained_evaluated_model";return
        requested=self.config.get("device","auto")
        self.device="cuda:0" if requested!="cpu" and torch.cuda.is_available() else "cpu"
        self.degraded=self.device=="cpu"
        for record in chosen:
            path=(ROOT/record["path"]).resolve()
            if not path.is_relative_to(ROOT) or not path.exists(): raise ValueError("registered_model_missing")
            if sha256(path)!=record["sha256"]: raise ValueError("model_hash_mismatch")
            if not record.get("run_id") or not record.get("initial_weights_sha256"): raise ValueError("model_provenance_missing")
            model=YOLO(str(path)); model.to(self.device)
            names=set(model.names.values())
            if names-set(CLASSES): raise ValueError("unapproved_model_class")
            active=set(record.get('active_classes',names))
            if not active or not active.issubset(names):raise ValueError('invalid_active_model_classes')
            if active & self.classes: raise ValueError("duplicate_class_models_not_supported")
            policy={'active_classes':sorted(active),'nms':record.get('inference',{}).get('nms',self.config.get('nms',True)),
                'confidence':float(record.get('inference',{}).get('confidence',self.config.get('confidence',.25)))}
            if not 0<policy['confidence']<1:raise ValueError('invalid_model_confidence')
            if not isinstance(policy['nms'],bool):raise ValueError('explicit_boolean_head_policy_required')
            # Warm every supported ready-batch size and common source aspect ratio.
            # A camera stopping changes the batch shape; its first CUDA kernels must
            # not be initialized while other cameras are waiting for observations.
            size=self.config.get("imgsz",640)
            batch_limit=self.config.get('microbatch',1) if self.device.startswith('cuda') else 1
            shapes=sorted({(size,size),(int(size*9/16),size),(int(size*3/4),size),(size,int(size*9/16)),(size,int(size*3/4))}) if self.device.startswith('cuda') else [(size,size)]
            for height,width in shapes:
                sample=np.zeros((height,width,3),dtype=np.uint8)
                for count in range(1,batch_limit+1):
                    model.predict([sample]*count,device=self.device,verbose=False,imgsz=size,rect=True,nms=policy['nms'],
                        quantize=16 if self.device.startswith("cuda") and self.config.get("half",True) else 32)
            self.actual_device=str(model.predictor.model.device)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
                if not self.actual_device.startswith("cuda"): raise RuntimeError("actual_gpu_execution_not_verified")
            self.models.append(model);self.policies.append(policy);self.hashes.append(record["sha256"]);self.classes.update(active)
            if record["evaluation_status"]!="accepted": self.degraded=True
        self.ready=True;self.provider="PyTorch CUDA" if self.actual_device.startswith("cuda") else "PyTorch CPU"
        if self.classes!=set(CLASSES): self.degraded=True

    def predict(self,frame):
        return self.predict_many([frame])[0]

    def predict_many(self,frames):
        output=[[] for _ in frames]
        for model,policy in zip(self.models,self.policies):
            display_threshold=policy['confidence']
            tracking_threshold=min(display_threshold,self.config.get('tracking_low_confidence',display_threshold)) if {'person','forklift'}&set(policy['active_classes']) else display_threshold
            results=model.predict([frame.image for frame in frames],device=self.device,imgsz=self.config.get("imgsz",640),conf=tracking_threshold,
                quantize=16 if self.device.startswith("cuda") and self.config.get("half",True) else 32,rect=True,nms=policy['nms'],verbose=False)
            if len(results)!=len(frames):raise RuntimeError("batch_result_count_mismatch")
            for index,(frame,result) in enumerate(zip(frames,results)):
                if tuple(result.orig_shape)!=(frame.source_height,frame.source_width):raise RuntimeError("batch_frame_shape_mismatch")
                # result.boxes.xyxy is already original image pixels.
                for row in result.boxes.data.cpu().numpy():
                    x1,y1,x2,y2,confidence,cls=row[:6]
                    name=result.names.get(int(cls))
                    if name not in policy['active_classes'] or not np.isfinite(confidence): continue
                    # Low-score person/forklift observations may associate with an
                    # existing track; do not lower PPE class detection thresholds.
                    if name not in {'person','forklift'} and confidence<display_threshold:continue
                    try: box=validate_box((x1,y1,x2,y2),frame.source_width,frame.source_height)
                    except ValueError: continue
                    output[index].append(Detection(name,box,float(confidence),last_observed_frame=frame.frame_id,last_observed_monotonic=frame.received_monotonic))
        return output

    def fallback_cpu(self):
        import torch
        for model in self.models: model.to("cpu")
        self.device=self.actual_device="cpu";self.provider="PyTorch CPU";self.degraded=True;self.error="gpu_failure_cpu_fallback"
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def status(self):
        return {"ready":self.ready,"provider":self.provider,"device":self.device,"actual_device":self.actual_device,
                "classes":sorted(self.classes),"missing_classes":sorted(set(CLASSES)-self.classes),"model_hashes":self.hashes,
                "degraded":self.degraded,"error":self.error,
                "model_policies":self.policies,
                "inference_cpu_threads":self.config.get('inference_cpu_threads',1),
                "nms_setting":self.config.get('nms'),"tracking_low_confidence":self.config.get('tracking_low_confidence',self.config.get('confidence',.25)),
                "active_head_modes":["one_to_one" if m.predictor.model.end2end else "one_to_many" for m in self.models if getattr(m,'predictor',None)]}
