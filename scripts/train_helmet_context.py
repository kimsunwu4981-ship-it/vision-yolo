"""Existing helmet labels plus explicit no-hat hard negatives, with context crops."""
import argparse,random,importlib.metadata
from pathlib import Path
import cv2,numpy as np,yaml
from app.common import ROOT,read_json,write_json,sha256,utc_now


def prepare(run_id):
    directory=ROOT/'data/derived'/run_id;manifests=ROOT/'data/manifests'/run_id
    assert not directory.exists() and not manifests.exists();directory.mkdir(parents=True);manifests.mkdir(parents=True)
    lower_train=read_json(ROOT/'data/manifests/body-parts-yolo26n-v1/train.json')['records']
    # Train-only ontology review of all 571 existing class-8 boxes. Exclude
    # ambiguous dark/cap/hood records; never convert user-video pixels to labels.
    excluded={471,745,746,842,851}
    approved={r['image_sha256'] for i,r in enumerate(lower_train) if i not in excluded}
    audit=ROOT/'outputs/far-cctv-fix-20261117/hat-audit-index.json';assert audit.exists()
    write_json(manifests/'ontology-review.json',dict(source_manifest_sha256=sha256(ROOT/'data/manifests/body-parts-yolo26n-v1/train.json'),
        existing_hat_crop_count=571,excluded_record_indices=sorted(excluded),review='Assistant visual review of existing training class-8 crops; selected industrial hard hats only. Small/occluded annotations remain a source-data limitation.',
        approved_train_source_hashes=sorted(approved),user_video_used=False))
    hashes={};stats={};rng=random.Random(73)
    for split in ('train','validation','test'):
        inputs={}
        for parent in ['ppe-yolo26s-balanced-v3','body-parts-yolo26n-v1']:
            for r in read_json(ROOT/'data/manifests'/parent/(split+'.json'))['records']:
                entries=[list(map(float,l.split())) for l in Path(r['source_label']).read_text().splitlines() if l.strip()]
                if any(int(e[0])==8 for e in entries) and (split!='train' or r['image_sha256'] not in approved):continue
                inputs[r['image_sha256']]=(r,entries)
        chosen=sorted(inputs.values(),key=lambda item:item[0]['image_sha256'])
        if split!='train':chosen=random.Random(74).sample(chosen,min(240,len(chosen)))
        images=directory/split/'images';labels=directory/split/'labels';images.mkdir(parents=True);labels.mkdir(parents=True);records=[];counts={'helmets':0,'negative_crops':0}
        def save(im,objects,row,transform,tag):
            n=len(records);image=images/f'{n:05d}-{tag}.jpg';label=labels/(image.stem+'.txt');h,w=im.shape[:2]
            ok,data=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,90]);assert ok;image.write_bytes(data.tobytes())
            label.write_text(''.join('0 '+' '.join(f'{v:.8f}' for v in box)+'\n' for box in objects),encoding='utf-8')
            records.append(dict(image=str(image.relative_to(ROOT)),label=str(label.relative_to(ROOT)),image_sha256=sha256(image),derived_label_sha256=sha256(label),split=split,
                source_records=[dict(image=row['source_image'],label=row['source_label'],image_sha256=row['image_sha256'],derived_label_sha256=row['source_label_sha256'],group=row['group'])],
                transform=transform,style=tag,user_video_used=False))
            counts['helmets']+=len(objects);counts['negative_crops']+=int(not objects and tag!='native')
        for row,entries in chosen:
            assert sha256(row['source_image'])==row['image_sha256'] and sha256(row['source_label'])==row['source_label_sha256']
            im=cv2.imdecode(np.fromfile(row['source_image'],np.uint8),cv2.IMREAD_COLOR);h,w=im.shape[:2]
            positive=[e[1:] for e in entries if int(e[0]) in {1,8,9}]
            save(im,positive,row,{'operation':'native'},'native')
            heads=[e for e in entries if int(e[0]) in {1,8,9,13}]
            for cls,x,y,bw,bh in heads:
                edge=min(max(w,h),max(64,round(max(bw*w,bh*h)*rng.choice([4,6,8]))))
                left=max(0,min(w-edge,round(x*w-edge/2)));top=max(0,min(h-edge,round(y*h-edge/2)));right=min(w,left+edge);bottom=min(h,top+edge)
                objects=[]
                for px,py,pw,ph in positive:
                    x1=max(left,(px-pw/2)*w);x2=min(right,(px+pw/2)*w);y1=max(top,(py-ph/2)*h);y2=min(bottom,(py+ph/2)*h)
                    if x2-x1>=2 and y2-y1>=2:objects.append([((x1+x2)/2-left)/(right-left),((y1+y2)/2-top)/(bottom-top),(x2-x1)/(right-left),(y2-y1)/(bottom-top)])
                crop=im[top:bottom,left:right].copy();transform=dict(operation='context_crop',xyxy=[left,top,right,bottom],source_class=int(cls),source_box=[x,y,bw,bh])
                save(crop,objects,row,transform,'nohat-negative' if int(cls)==13 else 'helmet-context')
                if split=='train' and (int(cls)==13 or rng.random()<.3):
                    small=cv2.resize(crop,(max(16,round(crop.shape[1]*.4)),max(16,round(crop.shape[0]*.4))),interpolation=cv2.INTER_AREA)
                    degraded=cv2.resize(small,(crop.shape[1],crop.shape[0]),interpolation=cv2.INTER_LINEAR)
                    save(degraded,objects,row,{**transform,'degraded_resolution':list(small.shape[:2])},'degraded-context')
        path=manifests/(split+'.json');write_json(path,dict(records=records,user_video_used_for_training=False,scope='Existing labels only; nohat boxes generate hard-negative helmet crops.'))
        hashes[split]=sha256(path);stats[split]=dict(rows=len(records),source_images=len(chosen),**counts)
        (directory/(split+'.txt')).write_text('\n'.join(str(ROOT/r['image']) for r in records)+'\n',encoding='utf-8')
    spec=directory/'data.yaml';spec.write_text(yaml.safe_dump(dict(path=str(directory),train='train.txt',val='validation.txt',test='test.txt',names=['helmet'])),encoding='utf-8')
    write_json(manifests/'selection.json',dict(statistics=stats,manifest_sha256=hashes))
    return spec,hashes,stats


def main(run_id,epochs):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(1);run=ROOT/'outputs/training'/run_id;assert not run.exists()
    data,hashes,stats=prepare(run_id);run.mkdir(parents=True)
    parent=read_json(ROOT/'models/ppe-yolo26s-v2.provenance.json');source=ROOT/'models/ppe-yolo26s-v2.pt';assert sha256(source)==parent['checkpoint_sha256']
    model=YOLO(str(source));head=model.model.model[-1]
    for branch in ('cv3','one2one_cv3'):
        for scale in range(3):
            old=getattr(head,branch)[scale][-1];new=torch.nn.Conv2d(old.in_channels,1,old.kernel_size,old.stride,old.padding,bias=True)
            new.weight.data.copy_(old.weight.data[1:2]);new.bias.data.copy_(old.bias.data[1:2]);getattr(head,branch)[scale][-1]=new
    head.nc=1;head.no=head.reg_max*4+1;model.model.names={0:'helmet'};model.model.yaml['nc']=1
    init=run/'helmet-initialization.pt';model.save(str(init));model=YOLO(str(init))
    params=dict(data=str(data),epochs=epochs,imgsz=640,batch=12,device=0,workers=0,seed=73,deterministic=True,optimizer='AdamW',lr0=.00015,
        warmup_bias_lr=.00015,warmup_epochs=1,weight_decay=.0001,freeze=10,nms=False,patience=epochs,project=str(run.parent),name=run.name,
        exist_ok=True,cache=False,plots=True,amp=True,mosaic=0.,mixup=0.,scale=.15,translate=.08,degrees=25,shear=2,perspective=.0002,fliplr=.5,close_mosaic=0,max_det=300,verbose=False)
    p=dict(run_id=run_id,status='running',started_at=utc_now(),parameters=params,manifest_sha256=hashes,
        manifest_paths={s:f'data/manifests/{run_id}/{s}.json' for s in hashes},statistics=stats,initial_checkpoint_sha256=sha256(source),
        initial_weights_sha256=parent['initial_weights_sha256'],initial_weights_source=parent['initial_weights_source'],parent_run_id=parent['run_id'],
        strategy='Preserve trained helmet class row; add reviewed existing industrial-hat labels, explicit nohat hard-negative crops and reduced-resolution context crops; train neck/head, freeze backbone statistics.',
        packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','ultralytics')},gpu=torch.cuda.get_device_name(0),
        user_video_used_for_training=False,external_dataset_used=False,field_acceptance=False)
    def lock(trainer):
        for i,layer in enumerate(trainer.model.model):
            if i<10:layer.eval()
    model.add_callback('on_train_batch_start',lock);write_json(run/'provenance.json',p)
    try:
        model.train(**params);best=run/'weights/best.pt';p.update(status='completed',completed_at=utc_now(),checkpoint=str(best.relative_to(ROOT)),checkpoint_sha256=sha256(best),checkpoint_selection='best existing-label context validation fitness')
    except Exception as error:p.update(status='failed',error_type=type(error).__name__);raise
    finally:write_json(run/'provenance.json',p)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='helmet-yolo26s-context-v6');parser.add_argument('--epochs',type=int,default=10);a=parser.parse_args();main(a.run_id,a.epochs)
