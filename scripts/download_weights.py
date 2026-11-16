"""Only official YOLO initialization weights. No external training data."""
import argparse
import urllib.request
from app.common import ROOT,sha256,write_json

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--model",choices=["yolo26s.pt","yolo26n.pt","yolo26n-pose.pt"],default="yolo26s.pt");a=p.parse_args()
    url="https://github.com/ultralytics/assets/releases/download/v8.4.0/"+a.model
    target=ROOT/"models"/a.model;target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        temporary=target.with_suffix(".download");urllib.request.urlretrieve(url,temporary);temporary.replace(target)
    write_json(ROOT/"models"/(a.model+".source.json"),{"url":url,"sha256":sha256(target),"role":"official_pretrained_initialization_only"})
    print(a.model,sha256(target))
