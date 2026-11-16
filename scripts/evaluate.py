from __future__ import annotations
import argparse
import os
import json
import shutil
from pathlib import Path
from app.common import ROOT,read_json,write_json,sha256,utc_now


def evaluate(run_id,register=False):
    os.environ["YOLO_CONFIG_DIR"]=str(ROOT/".cache/ultralytics");os.environ["YOLO_AUTOINSTALL"]="False"
    import torch
    from ultralytics import YOLO
    run=ROOT/"outputs/training"/run_id;provenance=read_json(run/"provenance.json")
    if not provenance or provenance.get("status")!="completed":raise ValueError("completed_new_run_required")
    checkpoint=ROOT/provenance["checkpoint"]
    if sha256(checkpoint)!=provenance["checkpoint_sha256"]:raise ValueError("checkpoint_changed")
    model=YOLO(str(checkpoint));reports={}
    for split in ("val","test"):
        metrics=model.val(data=str(ROOT/"data/derived/ppe_v2/data.yaml"),split=split,imgsz=provenance["parameters"]["imgsz"],
            batch=provenance["parameters"]["batch"],device=0 if torch.cuda.is_available() else "cpu",workers=0,
            project=str(run/"evaluation"),name=split,plots=True,verbose=False)
        classes={}
        for i,cls in enumerate(metrics.box.ap_class_index):
            precision,recall,ap50,ap=metrics.box.class_result(i)
            classes[model.names[int(cls)]]={"precision":float(precision),"recall":float(recall),"AP50":float(ap50),"AP50_95":float(ap)}
        reports[split]={"classes":classes,"aggregate":{k:float(v) for k,v in metrics.results_dict.items()},"images":len(read_json(ROOT/"data/manifests"/("validation.json" if split=="val" else "test.json"))["records"])}
    result={"run_id":run_id,"checkpoint_sha256":sha256(checkpoint),"evaluated_at":utc_now(),"splits":reports,
        "field_independent_test":False,"night_occlusion_event_metrics":"not evaluated: no annotated source videos",
        "small_object_metrics":"object-size annotations available; separate subset not yet evaluated",
        "forklift":"not trained: complete labeled source missing","deployment_acceptance":False}
    write_json(run/"evaluation.json",result)
    if register:
        if provenance.get("smoke"):raise ValueError("smoke_model_cannot_be_registered")
        target=ROOT/"models"/(run_id+".pt");shutil.copyfile(checkpoint,target)
        record={"run_id":run_id,"path":str(target.relative_to(ROOT)),"sha256":sha256(target),"classes":list(model.names.values()),
            "enabled":True,"evaluation_status":"development_evaluated","initial_weights_sha256":provenance["initial_weights_sha256"],
            "initial_weights_source":provenance["initial_weights_source"],"manifest_sha256":provenance["manifest_sha256"],
            "evaluation":result,"license":"AGPL-3.0; source dataset CC-BY-4.0 attribution also retained"}
        write_json(ROOT/"models/model_registry.json",{"schema_version":1,"models":[record],"status":"PPE development only; forklift and field acceptance unavailable"})
        write_json(ROOT/"models"/(run_id+".provenance.json"),provenance)
    print(json.dumps(result))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("run_id");p.add_argument("--register",action="store_true");a=p.parse_args();evaluate(a.run_id,a.register)
