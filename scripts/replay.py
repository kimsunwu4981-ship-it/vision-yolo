"""Exhaustive offline video analysis: every decoded frame, no latest-frame dropping."""
import argparse
import csv
import json
import platform
import time
import uuid
from dataclasses import asdict
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


def replay(paths,output):
    config=load_config();detector=Detector(config["runtime"]);detector.load()
    if not detector.ready:raise RuntimeError(detector.error)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write_json(output/"inputs_manifest.json",{"inputs":[{"camera_id":f"cam{i+1}","path":str(Path(p).resolve()),"sha256":sha256(p)} for i,p in enumerate(paths)]})
    write_json(output/"run_manifest.json",{"mode":"OFFLINE_EXHAUSTIVE","started_at":utc_now(),"config":config,"backend":detector.status(),"hardware":platform.platform(),"dropped_frames":0})
    tracker=TrackerBank();ppe=PPE();rules=Rules(config["rules"]);events=EventEngine()
    handles={name:open(output/(name+".jsonl"),"w",encoding="utf-8") for name in ["detections","tracks","events"]}
    perf=open(output/"performance.csv","w",newline="",encoding="utf-8");writer=csv.DictWriter(perf,fieldnames=["camera_id","session_id","frame_id","source_pts","processing_ms","rss_bytes"]);writer.writeheader()
    try:
        for i,path in enumerate(paths):
            cid=f"cam{i+1}";session=uuid.uuid4().hex
            with av.open(path) as container:
                stream=container.streams.video[0]
                for frame_id,raw in enumerate(container.decode(stream),1):
                    pts=float(raw.pts*raw.time_base) if raw.pts is not None else (frame_id-1)/float(stream.average_rate or 25)
                    frame=Frame(cid,session,frame_id,raw.width,raw.height,pts,utc_now(),pts,raw.to_ndarray(format="bgr24"))
                    start=time.perf_counter();detections=detector.predict(frame)
                    # Serialize raw detections BEFORE tracking annotates association fields.
                    handles["detections"].write(json.dumps({**frame.metadata(),"detections":[asdict(d) for d in detections]})+"\n")
                    tracks=tracker.update(frame,detections);states=ppe.update(frame,tracks,detector.classes)
                    handles["tracks"].write(json.dumps({**frame.metadata(),"tracks":[asdict(d) for d in tracks],"ppe":states})+"\n")
                    camera=config["cameras"][min(i,len(config["cameras"])-1)]
                    from app.common import read_json
                    zones=read_json(ROOT/"configs/zones"/(cid+".json"),[])
                    for event in events.update(frame,rules.update(frame,tracks,states,zones,camera),";".join(detector.hashes),config["revision"]):
                        handles["events"].write(json.dumps({**frame.metadata(),"event":event})+"\n")
                    writer.writerow({"camera_id":cid,"session_id":session,"frame_id":frame_id,"source_pts":pts,"processing_ms":(time.perf_counter()-start)*1000,"rss_bytes":psutil.Process().memory_info().rss})
    finally:
        for handle in handles.values():handle.close()
        perf.close()
    print(str(output))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("videos",nargs="+");p.add_argument("--output",required=True);a=p.parse_args();replay(a.videos,a.output)
