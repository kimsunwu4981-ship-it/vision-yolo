"""Diagnostic original-frame predictions, not training labels or recall ground truth."""
import argparse,time
from dataclasses import asdict
from collections import Counter
import av,cv2
from app.common import ROOT,write_json,sha256
from app.config import load_config
from app.inference import Detector
from app.orientation import decode_bgr
from app.types import Frame


def main(registry,output):
    detector=Detector(load_config()['runtime'],ROOT/registry);detector.load();out=ROOT/output;out.mkdir(exist_ok=True,parents=True);records=[]
    for source,tag,chosen in [(ROOT.parent.parent/'크레인 재해.mp4','accident',{1,100,199,298,397,496,596,695,794,893,992,1092}),
                              (ROOT.parent.parent/'크레인 점검.mp4','inspection',{1,100,200,300,400,500})]:
        with av.open(str(source)) as container:
            for index,raw in enumerate(container.decode(video=0),1):
                if index not in chosen:continue
                im,transform=decode_bgr(raw);h,w=im.shape[:2];pts=float(raw.pts*raw.time_base)
                frame=Frame('probe',tag,index,w,h,pts,'2026-11-17T00:00:00Z',pts,im,preprocess_transform=transform)
                start=time.perf_counter();detections=detector.predict(frame);elapsed=time.perf_counter()-start;canvas=im.copy()
                visible=[d for d in detections if d.confidence>=(.25 if d.class_name=='person' else .0)]
                for d in visible:
                    if d.class_name not in {'person','helmet','safety_vest'}:continue
                    x1,y1,x2,y2=map(round,d.xyxy);color=(0,230,255) if d.class_name=='person' else (0,255,0)
                    cv2.rectangle(canvas,(x1,y1),(x2,y2),color,2)
                    label=f'{d.class_name} {d.confidence:.2f}';cv2.putText(canvas,label,(x1,max(15,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
                ok,data=cv2.imencode('.jpg',canvas);assert ok;(out/f'{tag}-{index:04d}.jpg').write_bytes(data.tobytes())
                row=dict(source=source.name,source_sha256=sha256(source),frame=index,processing_s=elapsed,detections=[asdict(d) for d in detections]);records.append(row)
                print(tag,index,dict(Counter(d.class_name for d in visible)),round(elapsed,3),flush=True)
    write_json(out/'sample-predictions.json',dict(records=records,registry_sha256=sha256(ROOT/registry),diagnostic_only=True,accuracy_evaluated=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registry',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.registry,a.output)
