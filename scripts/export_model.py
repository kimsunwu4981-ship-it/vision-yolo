"""Optional target-machine export with a mandatory held-out equivalence report."""
import argparse
import os
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from app.common import ROOT,read_json,write_json,sha256

def export(run_id,fmt,device):
    from ultralytics import YOLO
    record=read_json(ROOT/"outputs/training"/run_id/"provenance.json")
    if not record or record.get("status")!="completed":raise ValueError("new_training_run_required")
    model=YOLO(str(ROOT/record["checkpoint"]));target=model.export(format=fmt,quantize=16 if fmt=="engine" else 32,device=device,imgsz=record["parameters"]["imgsz"])
    exported=YOLO(target,task="detect");examples=read_json(ROOT/"data/manifests/validation.json")["records"][:32];failures=[]
    from app.geometry import overlap_fraction
    for sample in examples:
        path=str(ROOT/sample["image"]);a=model.predict(path,conf=.25,device=device,verbose=False)[0];b=exported.predict(path,conf=.25,device=device,verbose=False)[0]
        aa=a.boxes.data.cpu().numpy();bb=b.boxes.data.cpu().numpy();cost=np.ones((len(aa),len(bb)))
        for i,x in enumerate(aa):
            for j,y in enumerate(bb):
                if int(x[5])!=int(y[5]):continue
                ix=max(0,min(x[2],y[2])-max(x[0],y[0]));iy=max(0,min(x[3],y[3])-max(x[1],y[1]));inter=ix*iy
                union=(x[2]-x[0])*(x[3]-x[1])+(y[2]-y[0])*(y[3]-y[1])-inter
                cost[i,j]=1-inter/max(union,1)
        matched=0
        if cost.size:
            rr,cc=linear_sum_assignment(cost);matched=sum(cost[i,j]<=.1 and abs(aa[i,4]-bb[j,4])<=.05 for i,j in zip(rr,cc))
        if matched!=len(aa) or matched!=len(bb):failures.append(sample["image_sha256"])
    write_json(Path(target).with_suffix('.equivalence.json'),{"format":fmt,"source_sha256":sha256(ROOT/record["checkpoint"]),"export_sha256":sha256(target),"images":len(examples),"failures":failures,"passed":not failures,"target_device":str(device)})
    if failures:raise RuntimeError("export_equivalence_failed")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("run_id");p.add_argument("--format",choices=["onnx","engine"],default="onnx");p.add_argument("--device",default="0");a=p.parse_args();export(a.run_id,a.format,a.device)
