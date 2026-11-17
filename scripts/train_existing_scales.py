"""Small-object adaptation using only previously labeled, split-isolated images.

User videos are never used for training. Reduced scenes and grids preserve all
source boxes except boxes smaller than two pixels after the recorded transform.
"""
import argparse,random,shutil,importlib.metadata
from collections import Counter
from pathlib import Path
import cv2,numpy as np,yaml
from app.common import ROOT,read_json,write_json,sha256,utc_now


def prepare(run_id,kind):
    directory=ROOT/'data/derived'/run_id;manifest_dir=ROOT/'data/manifests'/run_id
    if directory.exists() or manifest_dir.exists():raise ValueError('dataset_already_exists')
    rng=random.Random(61);parent=ROOT/'data/manifests/ppe-yolo26s-balanced-v3'
    directory.mkdir(parents=True);manifest_dir.mkdir(parents=True)
    hashes={};statistics={}
    for split in ('train','validation','test'):
        source_records=[{**r,'annotation_schema':'upper'} for r in read_json(parent/(split+'.json'))['records']]
        extra_manifest=None
        if kind=='ppe':
            # Lower-case PPE sources have no complete person annotations, so use
            # them only in this two-class PPE model. Ambiguous hat is excluded.
            extra_manifest=ROOT/'data/manifests/body-parts-yolo26n-v1'/(split+'.json')
            for r in read_json(extra_manifest)['records']:
                ids={int(line.split()[0]) for line in Path(r['source_label']).read_text().splitlines() if line.strip()}
                if 8 in ids:continue
                source_records.append({**r,'label':r['source_label'],'derived_label_sha256':r['source_label_sha256'],'annotation_schema':'lower'})
        # Selection happens within the established split; source IDs never move.
        source_records=sorted(source_records,key=lambda r:r['image_sha256'])
        chosen=source_records if split=='train' else random.Random(62).sample(source_records,min(240,len(source_records)))
        images=directory/split/'images';labels=directory/split/'labels';images.mkdir(parents=True);labels.mkdir(parents=True)
        cache={};records=[];counts=Counter();heights=[]
        def read(record):
            key=record['image_sha256']
            if key not in cache:
                path=ROOT/record['image'];label=ROOT/record['label']
                assert sha256(path)==key and sha256(label)==record['derived_label_sha256']
                im=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
                boxes=[]
                for line in label.read_text().splitlines():
                    cls,x,y,w,h=map(float,line.split())
                    if kind=='person' and int(cls)!=0:continue
                    if kind=='ppe':
                        mapping={9:0,15:1} if record['annotation_schema']=='lower' else {1:0,2:1}
                        if int(cls) not in mapping:continue
                        cls=mapping[int(cls)]
                    boxes.append([int(cls),x,y,w,h])
                cache[key]=(im,boxes)
            return cache[key]
        def save(name,canvas,objects,inputs,transforms,style):
            image=images/(name+'.jpg');label=labels/(name+'.txt')
            ok,data=cv2.imencode('.jpg',canvas,[cv2.IMWRITE_JPEG_QUALITY,92]);assert ok;image.write_bytes(data.tobytes())
            lines=[]
            for cls,x,y,w,h in objects:
                assert 0<x<1 and 0<y<1 and 0<w<=1 and 0<h<=1
                lines.append(str(cls)+' '+' '.join(f'{v:.8f}' for v in (x,y,w,h)));counts[str(cls)]+=1;heights.append(h*640)
            label.write_text('\n'.join(lines)+'\n',encoding='utf-8')
            records.append(dict(image=str(image.relative_to(ROOT)),label=str(label.relative_to(ROOT)),image_sha256=sha256(image),
                derived_label_sha256=sha256(label),split=split,source_records=[{k:r[k] for k in ('image','label','image_sha256','derived_label_sha256','group','annotation_schema')} for r in inputs],
                transforms=transforms,style=style,transform_version='existing-label-small-scenes-v1',synthetic_scene=True,
                user_video_used=False,independent_sample=False,license='CC-BY-4.0 attribution inherited from source manifests'))
        def place(canvas,record,edge,left,top):
            im,boxes=read(record);ih,iw=im.shape[:2];ratio=edge/max(ih,iw);nw,nh=round(iw*ratio),round(ih*ratio)
            resized=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_AREA)
            canvas[top:top+nh,left:left+nw]=resized;objects=[]
            for cls,x,y,w,h in boxes:
                if w*nw<2 or h*nh<2:continue
                objects.append([cls,(left+x*nw)/640,(top+y*nh)/640,w*nw/640,h*nh/640])
            return objects,dict(edge=edge,left=left,top=top,source_size=[iw,ih],placed_size=[nw,nh],discarded_under_2px=len(boxes)-len(objects))
        for n,record in enumerate(chosen):
            # Native-size scenes retain existing near-field recognition.
            canvas=np.full((640,640,3),114,np.uint8);objects,transform=place(canvas,record,640,0,0)
            save(f'native-{n:04d}',canvas,objects,[record],[transform],'native')
            # A smaller full scene, not a person-shaped synthetic patch.
            edge=rng.choice([96,128,160,224]);left=rng.randint(0,640-edge);top=rng.randint(0,640-edge)
            canvas=np.full((640,640,3),rng.randint(85,145),np.uint8);objects,transform=place(canvas,record,edge,left,top)
            save(f'shrink-{n:04d}',canvas,objects,[record],[transform],'shrunk')
        for n in range(len(chosen) if split=='train' else 120):
            side=rng.choice([3,4,5,6]);cell=640//side;canvas=np.full((640,640,3),114,np.uint8);objects=[];inputs=[];transforms=[]
            for row in range(side):
                for col in range(side):
                    record=rng.choice(chosen);edge=rng.randint(int(cell*.7),cell)
                    left=col*cell+rng.randint(0,cell-edge);top=row*cell+rng.randint(0,cell-edge)
                    boxes,transform=place(canvas,record,edge,left,top);objects.extend(boxes);inputs.append(record);transforms.append(transform)
            # Reproduce varied acquisition sharpness without generating labels.
            style='grid'
            if split=='train' and rng.random()<.3:
                if rng.random()<.5:
                    blurred=cv2.GaussianBlur(canvas,(0,0),1);canvas=cv2.addWeighted(canvas,1.8,blurred,-.8,0);style='grid-sharpened'
                else:canvas=cv2.GaussianBlur(canvas,(3,3),.65);style='grid-softened'
            save(f'grid-{n:04d}',canvas,objects,inputs,transforms,style)
        manifest=manifest_dir/(split+'.json');write_json(manifest,dict(records=records,parent_manifest_sha256=sha256(parent/(split+'.json')),
            additional_source_manifest_sha256=sha256(extra_manifest) if extra_manifest else None,
            limitation='Synthetic scale/scene variants within original source split; not independent field images.'))
        hashes[split]=sha256(manifest)
        (directory/(split+'.txt')).write_text('\n'.join(str(ROOT/r['image']) for r in records)+'\n',encoding='utf-8')
        statistics[split]=dict(images=len(records),source_images=len(chosen),object_counts=dict(counts),height_percentiles=np.percentile(heights,[10,50,90]).tolist())
    names=['person'] if kind=='person' else ['helmet','safety_vest']
    data=directory/'data.yaml';data.write_text(yaml.safe_dump(dict(path=str(directory),train='train.txt',val='validation.txt',test='test.txt',names=names)),encoding='utf-8')
    write_json(manifest_dir/'selection.json',dict(kind=kind,statistics=statistics,manifest_sha256=hashes,seed=61,user_video_used_for_training=False))
    return data,hashes,statistics


def train(args):
    from ultralytics import YOLO
    import torch
    run=ROOT/'outputs/training'/args.run_id
    if run.exists():raise ValueError('run_already_exists')
    data,hashes,statistics=prepare(args.run_id,args.kind)
    initial=(ROOT/args.initial).resolve();registry=read_json(ROOT/'models/model_registry.json')
    parent=next((m for m in registry['models'] if (ROOT/m['path']).resolve()==initial),None)
    if parent is None and initial.is_relative_to(ROOT/'models'):
        prior=read_json(initial.with_suffix('.provenance.json'),{})
        if prior.get('status')=='completed' and prior.get('checkpoint_sha256')==sha256(initial):
            for split,digest in prior['manifest_sha256'].items():
                manifest=ROOT/prior.get('manifest_paths',{}).get(split,f'data/manifests/{split}.json')
                assert sha256(manifest)==digest
            parent={**prior,'sha256':prior['checkpoint_sha256']}
    assert parent and sha256(initial)==parent['sha256'],'only_this_project_verified_checkpoint'
    run.mkdir(parents=True)
    params=dict(data=str(data),epochs=args.epochs,imgsz=640,batch=12,device=0,workers=0,seed=63,deterministic=True,
        optimizer='AdamW',lr0=.0001,warmup_bias_lr=.0001,warmup_epochs=1,weight_decay=.0001,freeze=0,nms=False,
        patience=8,project=str(run.parent),name=run.name,exist_ok=True,cache=False,plots=True,amp=True,
        mosaic=0.,mixup=0.,scale=.2,translate=.1,degrees=25,shear=3,perspective=.0003,fliplr=.5,close_mosaic=0,
        max_det=600,verbose=False)
    provenance=dict(run_id=args.run_id,status='running',started_at=utc_now(),parameters=params,manifest_sha256=hashes,
        manifest_paths={s:f'data/manifests/{args.run_id}/{s}.json' for s in hashes},statistics=statistics,
        initial_checkpoint=str(initial.relative_to(ROOT)),initial_checkpoint_sha256=sha256(initial),parent_run_id=parent['run_id'],
        initial_weights_sha256=parent['initial_weights_sha256'],initial_weights_source=parent['initial_weights_source'],
        strategy='Unfreeze all layers; mix native scenes with label-preserving shrunk scenes and grids, varied angle/scale/sharpness.',
        packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','ultralytics')},gpu=torch.cuda.get_device_name(0),
        user_video_used_for_training=False,external_dataset_used=False,field_acceptance=False,
        validation_scope='Native and synthetic-small images from existing validation groups; test/video never select checkpoints.')
    write_json(run/'provenance.json',provenance)
    try:
        YOLO(str(initial)).train(**params);best=run/'weights/best.pt'
        provenance.update(status='completed',completed_at=utc_now(),checkpoint=str(best.relative_to(ROOT)),checkpoint_sha256=sha256(best),
            checkpoint_selection='best mixed-validation fitness')
    except Exception as error:
        provenance.update(status='failed',error_type=type(error).__name__);raise
    finally:write_json(run/'provenance.json',provenance)
    print(args.run_id,provenance['status'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--kind',choices=['person','ppe'],required=True)
    parser.add_argument('--initial',required=True);parser.add_argument('--epochs',type=int,default=20)
    train(parser.parse_args())
