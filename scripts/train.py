from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from app.common import ROOT,sha256,write_json,utc_now,read_json


def train(args):
    os.environ["YOLO_CONFIG_DIR"]=str(ROOT/".cache/ultralytics")
    os.environ["YOLO_AUTOINSTALL"]="False"
    import torch
    import yaml
    from ultralytics import YOLO
    if args.initial not in {"yolo26s.pt","yolo26n.pt"}:raise ValueError("official_initial_candidates_only")
    # Download only the official published initial weights; never discover old checkpoints.
    initial=ROOT/"models"/args.initial
    if not initial.exists():raise FileNotFoundError("Run scripts/download_weights.py first")
    manifests={split:ROOT/"data/manifests"/(split+".json") for split in ("train","validation","test")}
    if any(not read_json(path,{}).get("records") for path in manifests.values()):raise RuntimeError("audited_dataset_required")
    data=ROOT/"data/derived/ppe_v2/data.yaml"
    if args.smoke:
        smoke=ROOT/"data/derived/smoke";smoke.mkdir(parents=True,exist_ok=True)
        for split in ("train","validation","test"):
            records=read_json(manifests[split])["records"][:24 if split=="train" else 8]
            (smoke/(split+".txt")).write_text("\n".join(str((ROOT/r["image"]).resolve()) for r in records),encoding="utf-8")
        (smoke/"data.yaml").write_text(yaml.safe_dump({"path":str(smoke),"train":"train.txt","val":"validation.txt","test":"test.txt","names":["person","helmet","safety_vest"]}),encoding="utf-8")
        data=smoke/"data.yaml"
    run_id=args.run_id or (("smoke" if args.smoke else "ppe")+"-"+utc_now().replace(":","").replace(".","-"))
    run=ROOT/"outputs/training"/run_id
    if run.exists():raise ValueError("run_id_already_exists")
    run.mkdir(parents=True)
    device=0 if torch.cuda.is_available() and args.device!="cpu" else "cpu"
    params=dict(data=str(data),epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,device=device,workers=0,seed=args.seed,deterministic=True,
        optimizer="AdamW",lr0=.001,patience=args.patience,project=str(run.parent),name=run.name,exist_ok=True,cache=False,plots=True,
        amp=device!="cpu",save=True,close_mosaic=3,verbose=False)
    provenance={"run_id":run_id,"status":"running","smoke":args.smoke,"started_at":utc_now(),"initial_weights":args.initial,
        "initial_weights_source":"https://github.com/ultralytics/assets/releases/download/v8.4.0/"+args.initial,
        "initial_weights_sha256":sha256(initial),"manifest_sha256":{k:sha256(p) for k,p in manifests.items()},"parameters":params,
        "packages":{x:importlib.metadata.version(x) for x in ("torch","torchvision","ultralytics","numpy","av")},
        "device":str(device),"gpu":torch.cuda.get_device_name(0) if device!="cpu" else None,"torch_cuda_runtime":torch.version.cuda,
        "python":platform.python_version(),"evaluation_scope":"development source-family split; field independence not verified"}
    write_json(run/"provenance.json",provenance)
    try:
        model=YOLO(str(initial));model.train(**params)
        best=run/"weights/best.pt"
        provenance.update(status="completed",completed_at=utc_now(),checkpoint=str(best.relative_to(ROOT)),checkpoint_sha256=sha256(best),
            checkpoint_selection="Ultralytics best validation fitness, never test score")
    except Exception as error:
        provenance.update(status="failed",error_type=type(error).__name__);raise
    finally:write_json(run/"provenance.json",provenance)
    print(json.dumps({"run_id":run_id,"checkpoint":str(best),"sha256":provenance["checkpoint_sha256"]}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--initial",default="yolo26s.pt");p.add_argument("--run-id");p.add_argument("--smoke",action="store_true")
    p.add_argument("--epochs",type=int,default=30);p.add_argument("--imgsz",type=int,default=640);p.add_argument("--batch",type=int,default=16)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--patience",type=int,default=8);p.add_argument("--device",default="auto")
    train(p.parse_args())
