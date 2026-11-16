"""Refine this project's body localizer on annotation-preserving person-context crops."""
import argparse,json,importlib.metadata
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image
import yaml
from app.common import ROOT,read_json,write_json,sha256,utc_now


def prepare(run_id,parent):
    data=ROOT/'data/derived'/run_id;manifests=ROOT/'data/manifests'/run_id
    if data.exists() or manifests.exists():raise ValueError('dataset_run_already_exists')
    data.mkdir(parents=True);manifests.mkdir(parents=True);hashes={};counts={}
    for split in ('train','validation','test'):
        source_manifest=ROOT/'data/manifests'/parent/(split+'.json');records=[]
        images=data/split/'images';labels=data/split/'labels';images.mkdir(parents=True);labels.mkdir(parents=True)
        for source in read_json(source_manifest)['records']:
            with Image.open(ROOT/source['image']) as opened:im=opened.convert('RGB')
            w,h=im.size;objects=[]
            for line in (ROOT/source['label']).read_text().splitlines():
                c,x,y,bw,bh=map(float,line.split());objects.append((int(c),np.array([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])))
            heads=[b for c,b in objects if c==0];torsos=[b for c,b in objects if c==1]
            costs=np.array([[np.linalg.norm((a[:2]+a[2:])/2-(b[:2]+b[2:])/2)/max(1,np.linalg.norm(b[2:]-b[:2])) for b in torsos] for a in heads])
            if not costs.size:continue
            rr,cc=linear_sum_assignment(costs)
            for number,(i,j) in enumerate(zip(rr,cc)):
                if costs[i,j]>1.5:continue
                head,torso=heads[i],torsos[j];hc=(head[:2]+head[2:])/2;tc=(torso[:2]+torso[2:])/2
                far=tc+1.6*(tc-hc)
                low=np.minimum(np.minimum(head[:2],torso[:2]),far);high=np.maximum(np.maximum(head[2:],torso[2:]),far)
                pad=np.maximum(12,(high-low)*.15);left,top=np.maximum(0,np.floor(low-pad)).astype(int);right,bottom=np.minimum([w,h],np.ceil(high+pad)).astype(int)
                cw,ch=right-left,bottom-top
                if min(cw,ch)<24:continue
                converted=[]
                for c,b in objects:
                    clipped=np.maximum(b[:2],[left,top]).tolist()+np.minimum(b[2:],[right,bottom]).tolist()
                    a,y,e,z=clipped
                    if e-a<3 or z-y<3:continue
                    converted.append([c,((a+e)/2-left)/cw,((y+z)/2-top)/ch,(e-a)/cw,(z-y)/ch])
                name=source['image_sha256'][:24]+f'-{number:03d}';image=images/(name+'.jpg');label=labels/(name+'.txt')
                im.crop((left,top,right,bottom)).save(image,quality=96)
                label.write_text('\n'.join(str(c)+' '+' '.join(f'{v:.8f}' for v in values) for c,*values in converted)+'\n',encoding='utf-8')
                records.append({'source_image':source['source_image'],'source_image_sha256':source['image_sha256'],'source_label_sha256':source['source_label_sha256'],
                    'image':str(image.relative_to(ROOT)),'image_sha256':sha256(image),'label':str(label.relative_to(ROOT)),'derived_label_sha256':sha256(label),
                    'crop_xyxy':[int(left),int(top),int(right),int(bottom)],'group':source['group'],'split':split,
                    'transform_version':'annotated-head-torso-context-crop-v1; all intersecting annotated parts clipped; source split unchanged','license':source['license']})
        path=manifests/(split+'.json');write_json(path,{'records':records,'parent_manifest_sha256':sha256(source_manifest)});hashes[split]=sha256(path);counts[split]=len(records)
        (data/(split+'.txt')).write_text('\n'.join(str((ROOT/r['image']).resolve()) for r in records)+'\n',encoding='utf-8')
    spec=data/'data.yaml';spec.write_text(yaml.safe_dump({'path':str(data),'train':'train.txt','val':'validation.txt','test':'test.txt','names':['head','torso']},allow_unicode=True),encoding='utf-8')
    write_json(manifests/'selection.json',{'images':counts,'parent_dataset':parent,'split_unchanged':True,'user_video_used_for_training':False,'manifest_sha256':hashes})
    print(json.dumps(counts),flush=True);return spec,hashes


def train(run_id,parent,epochs):
    run=ROOT/'outputs/training'/run_id
    if run.exists():raise ValueError('training_run_already_exists')
    data,hashes=prepare(run_id,parent)
    from ultralytics import YOLO
    import torch
    parent_provenance=read_json(ROOT/'outputs/training'/parent/'provenance.json');initial=ROOT/parent_provenance['checkpoint']
    assert sha256(initial)==parent_provenance['checkpoint_sha256']
    params=dict(data=str(data),epochs=epochs,imgsz=640,batch=16,device=0,workers=0,seed=52,deterministic=True,optimizer='AdamW',lr0=.0002,
        warmup_bias_lr=.0002,warmup_epochs=1,freeze=0,nms=False,patience=10,project=str(run.parent),name=run_id,exist_ok=True,amp=True,
        mosaic=.1,mixup=0,scale=.25,translate=.1,close_mosaic=5,plots=True,cache=False,verbose=False)
    run.mkdir(parents=True)
    provenance={'run_id':run_id,'status':'running','smoke':False,'started_at':utc_now(),'initial_weights':str(initial.relative_to(ROOT)),
        'initial_weights_sha256':sha256(initial),'initial_weights_source':'this project newly trained candidate '+parent,'parent_run':parent,
        'root_initial_weights_source':parent_provenance['initial_weights_source'],'root_initial_weights_sha256':parent_provenance['initial_weights_sha256'],
        'manifest_sha256':hashes,'manifest_paths':{s:f'data/manifests/{run_id}/{s}.json' for s in hashes},'parameters':params,
        'packages':{n:importlib.metadata.version(n) for n in ('torch','torchvision','ultralytics')},'gpu':torch.cuda.get_device_name(0),
        'user_video_used_for_training':False,'field_acceptance':False,'role':'body_part_localizer; not a PPE status classifier'}
    write_json(run/'provenance.json',provenance)
    try:
        YOLO(str(initial)).train(**params);best=run/'weights/best.pt'
        provenance.update(status='completed',completed_at=utc_now(),checkpoint=str(best.relative_to(ROOT)),checkpoint_sha256=sha256(best),checkpoint_selection='best validation fitness')
    except Exception as error:provenance.update(status='failed',error_type=type(error).__name__);raise
    finally:write_json(run/'provenance.json',provenance)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='body-parts-yolo26n-v2');p.add_argument('--parent',default='body-parts-yolo26n-v1');p.add_argument('--epochs',type=int,default=20);a=p.parse_args();train(a.run_id,a.parent,a.epochs)
