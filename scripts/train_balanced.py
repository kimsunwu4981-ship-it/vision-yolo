"""Train an independently initialized candidate after capping dominant train groups.

Validation/test images and the user's diagnostic video never enter training.
"""
import argparse,random,platform,importlib.metadata,json
from collections import defaultdict,Counter
from pathlib import Path
import yaml
from app.common import ROOT,read_json,write_json,sha256,utc_now


def run(args):
    output=ROOT/'outputs/training'/args.run_id
    if output.exists():raise ValueError('run_already_exists')
    from ultralytics import YOLO
    import torch
    source=ROOT/'data/manifests'
    records=read_json(source/'train.json')['records'];groups=defaultdict(list)
    for record in records:groups[record['group']].append(record)
    rng=random.Random(args.seed);selected=[]
    for group,items in sorted(groups.items()):
        items=sorted(items,key=lambda r:r['image_sha256'])
        selected.extend(rng.sample(items,min(args.group_cap,len(items))))
    selected.sort(key=lambda r:r['image_sha256'])
    directory=ROOT/'data/derived'/args.run_id;directory.mkdir(parents=True,exist_ok=True)
    manifests=ROOT/'data/manifests'/args.run_id;manifests.mkdir(parents=True,exist_ok=True)
    manifest_paths={}
    for split in ('train','validation','test'):
        items=selected if split=='train' else read_json(source/(split+'.json'))['records']
        target=manifests/(split+'.json')
        write_json(target,{'records':items,'split':split,'strategy':'train source-group cap only; original validation/test unchanged','source_manifest_sha256':sha256(source/(split+'.json'))})
        manifest_paths[split]=target
        (directory/(split+'.txt')).write_text('\n'.join(str((ROOT/r['image']).resolve()) for r in items)+'\n',encoding='utf-8')
    for split in ('validation','test'):
        assert {r['image_sha256'] for r in selected}.isdisjoint({r['image_sha256'] for r in read_json(source/(split+'.json'))['records']})
    data=directory/'data.yaml'
    data.write_text(yaml.safe_dump({'path':str(directory),'train':'train.txt','val':'validation.txt','test':'test.txt','names':['person','helmet','safety_vest']},allow_unicode=True),encoding='utf-8')
    initial=ROOT/'models/yolo26s.pt'
    assert sha256(initial)=='646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b'
    output.mkdir(parents=True)
    params=dict(data=str(data),epochs=args.epochs,imgsz=640,batch=16,device=0,workers=0,seed=args.seed,deterministic=True,
        optimizer='AdamW',lr0=args.lr0,patience=10,freeze=args.freeze,project=str(output.parent),name=output.name,exist_ok=True,
        cache=False,plots=True,amp=True,close_mosaic=10,mosaic=.5,scale=.25,save=True,verbose=False)
    provenance={'run_id':args.run_id,'status':'running','smoke':False,'started_at':utc_now(),'initial_weights':'yolo26s.pt',
        'initial_weights_source':'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt','initial_weights_sha256':sha256(initial),
        'manifest_sha256':{k:sha256(p) for k,p in manifest_paths.items()},'manifest_paths':{k:str(p.relative_to(ROOT)) for k,p in manifest_paths.items()},
        'parameters':params,'train_images_before':len(records),'train_images_after':len(selected),'group_cap':args.group_cap,
        'group_counts_before':dict(Counter(r['group'] for r in records)),'group_counts_after':dict(Counter(r['group'] for r in selected)),
        'packages':{n:importlib.metadata.version(n) for n in ('torch','torchvision','ultralytics','numpy','av')},
        'gpu':torch.cuda.get_device_name(0),'torch_cuda_runtime':torch.version.cuda,'python':platform.python_version(),
        'user_video_used_for_training':False,'evaluation_scope':'Original grouped development split; no claim of independent field acceptance'}
    write_json(output/'provenance.json',provenance)
    try:
        YOLO(str(initial)).train(**params)
        best=output/'weights/best.pt'
        provenance.update(status='completed',completed_at=utc_now(),checkpoint=str(best.relative_to(ROOT)),checkpoint_sha256=sha256(best),checkpoint_selection='best original-validation fitness; not diagnostic video or test score')
    except Exception as error:provenance.update(status='failed',error_type=type(error).__name__);raise
    finally:write_json(output/'provenance.json',provenance)
    print(json.dumps({'run_id':args.run_id,'images':len(selected),'checkpoint_sha256':sha256(best)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='ppe-yolo26s-balanced-v3');p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--group-cap',type=int,default=180);p.add_argument('--freeze',type=int,default=10);p.add_argument('--lr0',type=float,default=.0003);p.add_argument('--seed',type=int,default=43)
    run(p.parse_args())
