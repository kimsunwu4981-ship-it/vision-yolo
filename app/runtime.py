from __future__ import annotations
import threading
import time
import queue
from dataclasses import asdict
from pathlib import Path
import cv2
import numpy as np
from app.capture import Capture,LatestSlot
from app.common import ROOT,read_json,write_json,revision,utc_now
from app.events import EventEngine
from app.inference import Detector
from app.monitoring import Metrics,SystemMonitor
from app.ppe import PPE
from app.rules import Rules
from app.storage import Store
from app.tracking import TrackerBank
from app.types import Frame,RuleObservation


def render(frame,tracks,ppe,zones,width,model_ready):
    source_h,source_w=frame.image.shape[:2]
    scale=min(1.,width/source_w)
    image=cv2.resize(frame.image,(int(source_w*scale),int(source_h*scale)),interpolation=cv2.INTER_AREA) if scale<1 else frame.image.copy()
    h,w=image.shape[:2]
    for zone in zones:
        points=np.asarray([[x*w,y*h] for x,y in zone["points"]],np.int32)
        cv2.polylines(image,[points],True,(80,180,255),2)
    colors={"person":(235,195,60),"helmet":(80,220,160),"safety_vest":(90,220,240),"forklift":(70,130,250)}
    for obj in tracks:
        if not obj.observed:continue
        x1,y1,x2,y2=(int(v*scale) for v in obj.xyxy);color=colors[obj.class_name]
        text=f"{obj.class_name} {obj.confidence:.2f}"+(f" #{obj.track_id}" if obj.track_id is not None else "")
        if obj.key in ppe:
            text+=" H:"+ppe[obj.key]["helmet"]["state"]+" V:"+ppe[obj.key]["safety_vest"]["state"]
        cv2.rectangle(image,(x1,y1),(x2,y2),color,2);cv2.putText(image,text,(x1,max(16,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
    stamp=f"{frame.camera_id} / {frame.session_id[:8]} / frame {frame.frame_id}"
    cv2.putText(image,stamp,(12,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    if not model_ready:cv2.putText(image,"MODEL NOT READY - NO SAFETY ANALYSIS",(12,52),cv2.FONT_HERSHEY_SIMPLEX,.5,(50,150,255),1)
    ok,jpeg=cv2.imencode(".jpg",image,[cv2.IMWRITE_JPEG_QUALITY,80])
    if not ok:raise RuntimeError("jpeg_encode_failed")
    return jpeg.tobytes()


class Runtime:
    def __init__(self,config,output=None):
        self.config=config;self.lock=threading.RLock();self.stop_event=threading.Event()
        self.cameras={c["id"]:Capture(c) for c in config["cameras"]}
        self.metrics={cid:Metrics() for cid in self.cameras};self.latest={};self.render_slots={cid:LatestSlot() for cid in self.cameras}
        self.detector=Detector(config["runtime"]);self.tracker=TrackerBank(config["runtime"]["observation_ttl_s"])
        rules=config["rules"]
        self.ppe=PPE(rules["ppe_confirm_s"],rules["ppe_clear_s"],rules["max_observation_gap_s"],rules["min_person_height"])
        self.rules=Rules(rules);self.events=EventEngine(rules["max_observation_gap_s"],rules["cooldown_s"])
        self.health_events=EventEngine(max_gap_s=3,cooldown_s=5)
        self.store=Store(output or ROOT/"outputs/runtime",config["runtime"]["storage"])
        self.system=SystemMonitor();self.zones={cid:read_json(ROOT/"configs/zones"/(cid+".json"),[]) for cid in self.cameras}
        self.thread=threading.Thread(target=self.run,name="gpu-scheduler",daemon=True)
        self.renderer=threading.Thread(target=self.render_loop,name="preview-encoder",daemon=True)
        self.error=None;self.loading=True
        self.evidence_render=queue.Queue(maxsize=16)
        self.evidence_requested=set()
        self.scheduler_cursor=0

    @property
    def config_revision(self):return revision({"base":self.config["revision"],"zones":self.zones})

    def start(self):
        self.store.thread.start();self.system.thread.start();self.renderer.start();self.thread.start()
        for capture in self.cameras.values():
            if capture.config.get("enabled"):capture.start()

    def close(self):
        self.stop_event.set()
        for capture in list(self.cameras.values()):capture.stop()
        self.thread.join(timeout=10);self.renderer.join(timeout=5);self.system.stop.set();self.system.thread.join(timeout=3);self.store.close()

    def add_camera(self,config):
        with self.lock:
            if config["id"] in self.cameras:raise ValueError("camera_id_exists")
            if len(self.cameras)>=4:raise ValueError("four_camera_limit")
            cid=config["id"];self.cameras[cid]=Capture(config);self.metrics[cid]=Metrics();self.render_slots[cid]=LatestSlot();self.zones[cid]=[]
            import yaml
            path=ROOT/"configs/cameras.local.yaml"
            path.write_text(yaml.safe_dump({"cameras":[c.config for c in self.cameras.values()]},allow_unicode=True),encoding="utf-8")
            self.config["revision"]=revision([c.config for c in self.cameras.values()])
        if config.get("enabled"):self.cameras[cid].start()

    def set_zones(self,camera_id,zones):
        with self.lock:
            write_json(ROOT/"configs/zones"/(camera_id+".json"),zones);self.zones[camera_id]=zones
        return self.config_revision

    def camera_status(self):
        output=[]
        for cid,capture in list(self.cameras.items()):
            metrics=self.metrics[cid].snapshot();age=metrics["last_observation_age_s"]
            status=capture.status
            if status=="connected" and capture.last_received is not None and time.monotonic()-capture.last_received>2:status="stale"
            output.append({"id":cid,"name":capture.config["name"],"mode":capture.mode,"status":status,"error":capture.error,
                "configured":bool(capture.source()),"session_id":capture.session_id,"zones":len(self.zones[cid]),
                "analysis_live":self.detector.ready and status=="connected" and age is not None and age<self.config["runtime"]["observation_ttl_s"],
                **capture.metrics(),**metrics})
        return output

    def snapshot(self):
        return {"time":utc_now(),"config_revision":self.config_revision,"cameras":self.camera_status(),"model":self.detector.status(),
            "system":self.system.value,"storage":{"error":self.store.error,"queued":self.store.jobs.qsize(),"dropped":self.store.dropped,
            "compressed_buffer_bytes":self.store.buffer_bytes},"loading":self.loading,"error":self.error}

    def run(self):
        try:self.detector.load()
        except Exception as error:self.detector.error=type(error).__name__;self.error="model_load_failed"
        self.loading=False;last_health=0
        while not self.stop_event.is_set():
            worked=False;batch=[];camera_items=list(self.cameras.items())
            order=camera_items[self.scheduler_cursor:]+camera_items[:self.scheduler_cursor]
            for cid,capture in order:
                if self.stop_event.is_set():break
                frame=capture.slot.take()
                if frame is None:continue
                if frame.session_id!=capture.session_id:continue
                if time.monotonic()-frame.received_monotonic>2:continue
                batch.append((cid,capture,frame))
                self.scheduler_cursor=(camera_items.index((cid,capture))+1)%max(1,len(camera_items))
                if len(batch)>=self.config["runtime"].get("microbatch",1):break
            start=time.perf_counter()
            try:
                predictions=self.detector.predict_many([f for _,_,f in batch]) if batch and self.detector.ready else [[] for _ in batch]
            except Exception as error:
                self.error="inference_"+type(error).__name__
                if self.detector.device.startswith("cuda"):
                    try:self.detector.fallback_cpu()
                    except Exception:self.detector.ready=False;self.detector.error="cpu_fallback_failed"
                else:self.detector.ready=False;self.detector.error="inference_failed"
                predictions=[[] for _ in batch]
                for cid,_,_ in batch:
                    for event in self.events.suspend_camera(cid):self.store.save_event(event)
                batch=[]
            for (cid,capture,frame),detections in zip(batch,predictions):
                worked=True;tracks=[];ppe={};changes=[]
                try:
                    if self.detector.ready:
                        tracks=self.tracker.update(frame,detections)
                        ppe=self.ppe.update(frame,tracks,self.detector.classes)
                        observations=self.rules.update(frame,tracks,ppe,self.zones[cid],capture.config)
                        current_revision=self.config_revision
                        for observation in observations:observation.key+=":"+current_revision
                        changes=self.events.update(frame,observations,";".join(self.detector.hashes),current_revision)
                        values=[item["state"] for person in ppe.values() for item in person.values()]
                        self.metrics[cid].ppe_unknown=sum(s=="unknown" for s in values)/len(values) if values else 0.
                        self.metrics[cid].record(frame,(time.perf_counter()-start)*1000)
                    for event in changes:
                        self.store.save_event(event)
                        if event["status"]=="active" and event["id"] not in self.evidence_requested:
                            try:
                                self.evidence_render.put_nowait((frame,tracks,ppe,list(self.zones[cid]),[event]))
                                self.evidence_requested.add(event["id"])
                            except queue.Full:self.store.error="evidence_render_queue_full"
                    if len(self.evidence_requested)>10000:self.evidence_requested={e["id"] for e in self.events.states.values()}
                    self.render_slots[cid].put((frame,tracks,ppe,list(self.zones[cid]),[]))
                except Exception as error:
                    self.error=type(error).__name__
                    for event in self.events.suspend_camera(cid):self.store.save_event(event)
            now=time.monotonic()
            if now-last_health>1:
                self.check_health();last_health=now
            if not worked:self.stop_event.wait(.003)

    def check_health(self):
        for cid,capture in list(self.cameras.items()):
            status=capture.status;offline=status in {"offline","stale"} or (status=="connected" and capture.last_received is not None and time.monotonic()-capture.last_received>2)
            if status!="connected" or offline:
                for event in self.events.suspend_camera(cid):self.store.save_event(event)
            frame=Frame(cid,"system",0,1,1,None,utc_now(),time.monotonic(),np.zeros((1,1,3),np.uint8))
            observations=[RuleObservation(cid+":offline","camera_offline",offline if capture.source() and status!="stopped" else None,
                basis={"connection":status},confirm_s=1.,clear_s=1.),
                RuleObservation(cid+":degraded","inference_degraded",self.detector.degraded or not self.detector.ready or self.error is not None,
                basis=self.detector.status(),confirm_s=0,clear_s=1.)]
            for event in self.health_events.update(frame,observations,";".join(self.detector.hashes),self.config_revision):self.store.save_event(event)

    def render_loop(self):
        last={}
        while not self.stop_event.is_set():
            worked=False
            try:
                frame,tracks,ppe,zones,changes=self.evidence_render.get_nowait()
                jpeg=render(frame,tracks,ppe,zones,self.config["runtime"]["preview_width"],self.detector.ready)
                for event in changes:self.store.save_event(event,frame,jpeg)
                self.evidence_render.task_done();worked=True
            except queue.Empty:pass
            except Exception as error:self.store.error="evidence_render_"+type(error).__name__
            for cid,slot in list(self.render_slots.items()):
                now=time.monotonic()
                if now<last.get(cid,0):continue
                item=slot.take()
                if item is None:continue
                worked=True;frame,tracks,ppe,zones,changes=item
                if frame.session_id!=self.cameras[cid].session_id:continue
                try:
                    jpeg=render(frame,tracks,ppe,zones,self.config["runtime"]["preview_width"],self.detector.ready)
                    self.latest[cid]=(frame,jpeg);self.metrics[cid].preview.append(time.monotonic())
                    last[cid]=max(now,last.get(cid,now))+1/self.config["runtime"]["preview_fps"]
                    self.metrics[cid].latency.append((time.monotonic()-frame.received_monotonic)*1000)
                    self.store.add_frame(frame,jpeg)
                    for event in changes:self.store.save_event(event,frame,jpeg)
                except Exception as error:self.error="render_or_evidence_"+type(error).__name__
            if not worked:self.stop_event.wait(.005)

    def preview(self,camera_id):
        item=self.latest.get(camera_id);capture=self.cameras[camera_id]
        if item:
            frame,jpeg=item
            if frame.session_id==capture.session_id and capture.status=="connected" and time.monotonic()-frame.received_monotonic<self.config["runtime"]["observation_ttl_s"]:
                return frame.key,jpeg
        image=np.zeros((360,640,3),np.uint8);image[:]=(25,21,17)
        cv2.putText(image,camera_id+" / "+capture.status.upper(),(30,170),cv2.FONT_HERSHEY_SIMPLEX,.8,(160,175,190),1,cv2.LINE_AA)
        cv2.putText(image,"NO CURRENT OBSERVATION",(30,210),cv2.FONT_HERSHEY_SIMPLEX,.5,(100,120,140),1)
        return (camera_id,capture.status,int(time.monotonic())),cv2.imencode(".jpg",image)[1].tobytes()
