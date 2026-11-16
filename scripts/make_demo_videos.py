"""Locally create clearly labeled STILL-IMAGE sequences for transport tests, not accuracy."""
from pathlib import Path
import argparse
import av
import cv2
import numpy as np
import yaml
from app.common import ROOT,read_json,write_json,sha256


def make(seconds=8,width=2560,height=1440,fps=25):
    records=read_json(ROOT/"data/manifests/test.json")["records"]
    output=ROOT/"outputs/demo";output.mkdir(parents=True,exist_ok=True)
    cameras=[];inputs=[]
    for index in range(4):
        record=records[index*17];path=ROOT/record["image"]
        image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError("image_decode_failed")
        scale=min(width/image.shape[1],height/image.shape[0]);iw=int(image.shape[1]*scale);ih=int(image.shape[0]*scale)
        image=cv2.resize(image,(iw,ih));base=np.zeros((height,width,3),np.uint8)
        ox=(width-iw)//2;oy=(height-ih)//2;base[oy:oy+ih,ox:ox+iw]=image
        target=output/f"demo{index+1}.mp4"
        with av.open(str(target),"w") as container:
            stream=container.add_stream("libx264",rate=fps);stream.width=width;stream.height=height;stream.pix_fmt="yuv420p";stream.options={"preset":"ultrafast","crf":"28"}
            for j in range(seconds*fps):
                frame=base.copy()
                cv2.rectangle(frame,(0,0),(width,70),(15,25,40),-1)
                cv2.putText(frame,f"DEMO {index+1} - STILL IMAGE TRANSPORT FIXTURE - frame {j}",(25,48),cv2.FONT_HERSHEY_SIMPLEX,1,(255,220,100),2)
                video=av.VideoFrame.from_ndarray(frame,format="bgr24")
                for packet in stream.encode(video):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
        cameras.append({"id":f"cam{index+1}","name":f"DEMO {index+1} · 정지 사진 전송 시험","source_type":"file","path":str(target),"enabled":True,"loop":True,"demo":True})
        inputs.append({"output":str(target),"sha256":sha256(target),"source_image_sha256":record["image_sha256"],"source_image":record["source_image"],"synthetic_temporal_sequence":True,"real_camera_video":False,"fps":fps,"resolution":[width,height]})
    (ROOT/"configs/cameras.demo.local.yaml").write_text(yaml.safe_dump({"cameras":cameras},allow_unicode=True),encoding="utf-8")
    write_json(output/"inputs_manifest.json",{"purpose":"transport and GPU functional tests only; no temporal accuracy claims","inputs":inputs})
    print("configs/cameras.demo.local.yaml")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seconds",type=int,default=8);a=p.parse_args();make(a.seconds)
