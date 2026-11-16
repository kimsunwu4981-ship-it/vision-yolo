"""Object-size slices and failure examples; no fabricated night/occlusion labels."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from app.common import ROOT,read_json,write_json


def evaluate(run_id):
    from ultralytics import YOLO
    import torch
    run=ROOT/"outputs/training"/run_id;record=read_json(run/"provenance.json");model=YOLO(str(ROOT/record["checkpoint"]))
    samples=read_json(ROOT/"data/manifests/test.json")["records"]
    counters={size:{name:{"ground_truth":0,"matched_at_iou_0.5":0} for name in ["person","helmet","safety_vest"]} for size in ["small","medium","large"]}
    failures=[];output=run/"evaluation/failures";output.mkdir(parents=True,exist_ok=True)
    for index,sample in enumerate(samples):
        path=ROOT/sample["image"];result=model.predict(str(path),conf=.25,imgsz=640,device=0 if torch.cuda.is_available() else "cpu",verbose=False)[0]
        h,w=result.orig_shape;truth=[]
        for line in (ROOT/sample["label"]).read_text().splitlines():
            cls,x,y,bw,bh=map(float,line.split());box=np.array([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]);size="small" if bw*w*bh*h<32**2 else ("medium" if bw*w*bh*h<96**2 else "large")
            truth.append((int(cls),box,size));counters[size][model.names[int(cls)]]["ground_truth"]+=1
        predictions=result.boxes.data.cpu().numpy();cost=np.ones((len(truth),len(predictions)))
        for i,(cls,a,size) in enumerate(truth):
            for j,b in enumerate(predictions):
                if int(b[5])!=cls:continue
                inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
                union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
                cost[i,j]=1-inter/max(union,1)
        matches=[]
        if cost.size:
            rr,cc=linear_sum_assignment(cost);matches=[(int(i),int(j)) for i,j in zip(rr,cc) if cost[i,j]<=.5]
        for i,j in matches:
            cls,box,size=truth[i];counters[size][model.names[cls]]["matched_at_iou_0.5"]+=1
        if len(matches)<len(truth) or len(matches)<len(predictions):
            failures.append({"image_sha256":sample["image_sha256"],"source":sample["source_image"],"missed":len(truth)-len(matches),"unmatched_predictions":len(predictions)-len(matches)})
            if len(failures)<=24:result.save(str(output/(sample["image_sha256"][:16]+".jpg")))
    for entries in counters.values():
        for metrics in entries.values():metrics["recall_at_conf_0.25_iou_0.5"]=metrics["matched_at_iou_0.5"]/metrics["ground_truth"] if metrics["ground_truth"] else None
    report={"object_size_pixels":"original 640x640 dataset pixels; <32^2 small, <96^2 medium","counts":counters,"failures":failures,"night":"unannotated/not evaluated","occlusion":"unannotated/not evaluated","evaluation_scope":"development held-out images"}
    write_json(run/"evaluation/slices.json",report);print(json.dumps({"failures":len(failures),"counts":counters}))

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("run_id");evaluate(p.parse_args().run_id)
