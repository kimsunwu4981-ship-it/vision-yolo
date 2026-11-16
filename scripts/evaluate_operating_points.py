"""Report fixed operating points on the existing development image test split.

No threshold search or model selection occurs here. Images are not field acceptance.
"""
import argparse
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment
from app.common import ROOT,read_json,write_json,sha256


def evaluate(output):
    from ultralytics import YOLO
    samples=read_json(ROOT/'data/manifests/test.json')['records']
    names=['person','helmet','safety_vest'];registry=read_json(ROOT/'models/model_registry.json')
    profiles={'before':[{'path':'models/ppe-yolo26s-v2.pt','active_classes':names,'inference':{'confidence':.25,'nms':True}}],
        'after':[r for r in registry['models'] if r.get('enabled')]}
    predictions={profile:[[] for _ in samples] for profile in profiles}
    model_info={}
    for profile,records in profiles.items():
        model_info[profile]=[]
        for record in records:
            path=ROOT/record['path'];policy=record['inference'];model=YOLO(str(path))
            model_info[profile].append({'sha256':sha256(path),'policy':policy,'active_classes':record['active_classes']})
            for start in range(0,len(samples),16):
                batch=samples[start:start+16]
                results=model.predict([str(ROOT/r['image']) for r in batch],conf=policy['confidence'],nms=policy['nms'],
                    imgsz=640,rect=True,device=0,quantize=16,verbose=False)
                for j,result in enumerate(results):
                    for row in result.boxes.data.cpu().numpy():
                        name=result.names[int(row[5])]
                        if name in record['active_classes']:
                            predictions[profile][start+j].append((name,row[:4].tolist(),float(row[4])))
        print(profile+' predictions complete',flush=True)
    def iou(a,b):
        overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap
        return overlap/max(union,1e-9)
    from PIL import Image
    counts={p:{name:Counter(tp=0,fp=0,fn=0) for name in names} for p in profiles}
    for index,sample in enumerate(samples):
        with Image.open(ROOT/sample['image']) as image:w,h=image.size
        truth=[]
        for line in (ROOT/sample['label']).read_text().splitlines():
            cls,x,y,bw,bh=map(float,line.split());truth.append((names[int(cls)],[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]))
        for profile in profiles:
            for name in names:
                expected=[r[1] for r in truth if r[0]==name];actual=[r[1] for r in predictions[profile][index] if r[0]==name]
                cost=np.array([[1-iou(a,b) if iou(a,b)>=.5 else 1000 for b in actual] for a in expected]).reshape(len(expected),len(actual))
                rr,cc=linear_sum_assignment(cost);tp=sum(cost[r,c]<1 for r,c in zip(rr,cc))
                counts[profile][name].update(tp=int(tp),fp=len(actual)-int(tp),fn=len(expected)-int(tp))
    metrics={}
    for profile,classes in counts.items():
        metrics[profile]={name:{**c,'precision':c['tp']/max(1,c['tp']+c['fp']),'recall':c['tp']/max(1,c['tp']+c['fn'])} for name,c in classes.items()}
    report={'images':len(samples),'test_manifest_sha256':sha256(ROOT/'data/manifests/test.json'),
        'iou':.5,'precision_bits':16,'metrics':metrics,'models':model_info,'selection_performed':False,
        'scope':'Fixed image operating points; weak track-association observations excluded; not field acceptance or event accuracy.'}
    write_json(output,report);print(metrics)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);evaluate(p.parse_args().output)
