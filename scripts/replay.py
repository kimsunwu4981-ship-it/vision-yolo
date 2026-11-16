"""Exhaustive offline video analysis: every decoded frame, no latest-frame dropping."""
import argparse
import csv
import json
import platform
import time
import uuid
from dataclasses import asdict
from datetime import datetime,timezone,timedelta
from pathlib import Path
import av
import psutil
from app.common import ROOT,sha256,write_json,utc_now
from app.config import load_config
from app.inference import Detector
from app.tracking import TrackerBank
from app.ppe import PPE
from app.rules import Rules
from app.events import EventEngine
from app.types import Frame
from app.monitoring import SystemMonitor
from app.orientation import decode_bgr


def replay(paths,output,source_mode='REPLAY',camera_file=None,no_zones=False):
    config=load_config(camera_file);detector=Detector(config["runtime"]);detector.load()
    if not detector.ready:raise RuntimeError(detector.error)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write_json(output/"inputs_manifest.json",{"inputs":[{"camera_id":f"cam{i+1}","path":str(Path(p).resolve()),"sha256":sha256(p)} for i,p in enumerate(paths)]})
    monitor=SystemMonitor();monitor.thread.start()
    time.sleep(.1)
    write_json(output/"run_manifest.json",{"mode":"OFFLINE_EXHAUSTIVE","source_mode":source_mode,"started_at":utc_now(),"config":config,"backend":detector.status(),"hardware":monitor.value or {"os":platform.platform()},"dropped_frames":0,
        "zones_disabled":no_zones,"entrypoint_sha256":sha256(__file__),
        "observation_clock":"virtual source PTS; received timestamps use replay origin plus PTS, actual decode time is in preprocess_transform",
        "source_code_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'app').glob('*.py'))}})
    settings=config['rules'];tracker=TrackerBank(config['runtime']['observation_ttl_s'],config['runtime'].get('confidence',.25),config['runtime'].get('tracking_low_confidence',.1))
    ppe=PPE(settings['ppe_confirm_s'],settings['ppe_clear_s'],settings['max_observation_gap_s'],settings['min_person_height'])
    rules=Rules(settings);events=EventEngine(settings['max_observation_gap_s'],settings['cooldown_s'])
    handles={name:open(output/(name+".jsonl"),"w",encoding="utf-8") for name in ["detections","tracks","events"]}
    perf=open(output/"performance.csv","w",newline="",encoding="utf-8");writer=csv.DictWriter(perf,fieldnames=["camera_id","session_id","frame_id","source_pts","processing_ms","rss_bytes","system_ram_gb","global_vram_gb"]);writer.writeheader()
    counts={}
    try:
        for i,path in enumerate(paths):
            cid=f"cam{i+1}";session=uuid.uuid4().hex
            origin=time.monotonic();origin_utc=datetime.now(timezone.utc);counts[cid]=0
            with av.open(path) as container:
                stream=container.streams.video[0]
                for frame_id,raw in enumerate(container.decode(stream),1):
                    pts=float(raw.pts*raw.time_base) if raw.pts is not None else (frame_id-1)/float(stream.average_rate or 25)
                    pixels,transform=decode_bgr(raw)
                    transform.update({"received_timestamp_stage":"replay_virtual_source_clock","decoded_at_utc":utc_now(),"decoded_monotonic":time.monotonic()})
                    height,width=pixels.shape[:2]
                    frame=Frame(cid,session,frame_id,width,height,pts,(origin_utc+timedelta(seconds=pts)).isoformat(),origin+pts,pixels,preprocess_transform=transform,source_mode=source_mode)
                    start=time.perf_counter();detections=detector.predict(frame)
                    # Serialize raw detections BEFORE tracking annotates association fields.
                    handles["detections"].write(json.dumps({**frame.metadata(),"detections":[asdict(d) for d in detections]})+"\n")
                    tracks=tracker.update(frame,detections);states=ppe.update(frame,tracks,detector.classes|detector.auxiliary_classes)
                    handles["tracks"].write(json.dumps({**frame.metadata(),"tracks":[asdict(d) for d in tracks],"ppe":states})+"\n")
                    camera=config["cameras"][min(i,len(config["cameras"])-1)]
                    from app.common import read_json
                    zones=[] if no_zones else read_json(ROOT/"configs/zones"/(cid+".json"),[])
                    for event in events.update(frame,rules.update(frame,tracks,states,zones,camera),";".join(detector.hashes),config["revision"]):
                        event['source_mode']=source_mode
                        handles["events"].write(json.dumps({**frame.metadata(),"event":event})+"\n")
                    writer.writerow({"camera_id":cid,"session_id":session,"frame_id":frame_id,"source_pts":pts,"processing_ms":(time.perf_counter()-start)*1000,"rss_bytes":psutil.Process().memory_info().rss,
                        "system_ram_gb":monitor.value.get('ram_used_gb'),"global_vram_gb":(monitor.value.get('gpu') or {}).get('vram_used_gb')})
                    counts[cid]+=1
    finally:
        for handle in handles.values():handle.close()
        perf.close()
        monitor.stop.set();monitor.thread.join(timeout=3)
    write_json(output/'summary.json',{'completed':True,'frames':counts,'dropped_frames':0,'accuracy_evaluated':False,'source_mode':source_mode})
    print(str(output))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("videos",nargs="+");p.add_argument("--output",required=True);p.add_argument('--source-mode',choices=['REPLAY','DEMO'],default='REPLAY');p.add_argument('--cameras');p.add_argument('--no-zones',action='store_true');a=p.parse_args();replay(a.videos,a.output,a.source_mode,a.cameras,a.no_zones)
