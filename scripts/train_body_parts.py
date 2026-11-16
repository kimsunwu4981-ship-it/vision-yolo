"""Train anatomical localization from existing reviewed head/clothing boxes only."""
import argparse,csv,json,random,hashlib,shutil,importlib.metadata
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import yaml
from app.common import ROOT,read_json,write_json,sha256,utc_now
from scripts.prepare_data import UnionFind


def prepare(run_id,cap=400):
    out=ROOT/'data/derived'/run_id;manifests=ROOT/'data/manifests'/run_id
    if out.exists() or manifests.exists():raise ValueError('dataset_run_already_exists')
    inventory=list(csv.DictReader((ROOT/'data/manifests/dataset_inventory.csv').open(encoding='utf-8-sig')))
    mapping={8:0,9:0,13:0,14:1,15:1};rows=[]
    for row in inventory:
        objects=json.loads(row.get('objects') or '[]');ids={int(o[0]) for o in objects}
        if ids&{8,9,13} and ids&{14,15} and not ids&{0,1,2,3,4}:
            row['parsed']=objects;rows.append(row)
    uf=UnionFind(len(rows));owners={}
    for i,r in enumerate(rows):
        for tag,value in [('hash',r['sha256']),('family',r['source_group']),('stem',Path(r['path']).stem.split('.rf.')[0])]:
            key=(tag,value)
            if key in owners:uf.union(i,owners[key])
            else:owners[key]=i
    hashes=np.array([int(r['dhash'],16) for r in rows],dtype=np.uint64);pop=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)
    for i in range(len(rows)):
        distances=pop[(hashes[i+1:]^hashes[i]).view(np.uint8).reshape(-1,8)].sum(axis=1)
        for j in np.flatnonzero(distances<=4):uf.union(i,i+1+int(j))
    previous={r['image_sha256']:split for split in ('train','validation','test') for r in read_json(ROOT/'data/manifests'/(split+'.json'))['records']}
    known=[r for r in inventory if r['sha256'] in previous]
    old_hashes=np.array([int(r['dhash'],16) for r in known],dtype=np.uint64)
    pins=defaultdict(set);groups=defaultdict(list)
    for i,row in enumerate(rows):
        key=uf.find(i);groups[key].append(row)
        distances=pop[(old_hashes^hashes[i]).view(np.uint8).reshape(-1,8)].sum(axis=1)
        pins[key].update(previous[known[int(j)]['sha256']] for j in np.flatnonzero(distances<=4))
    parts={s:[] for s in ('train','validation','test')};ratios={'train':.7,'validation':.15,'test':.15}
    seen=set();group_info=[];excluded=[];rng=random.Random(51)
    for key,group in sorted(groups.items(),key=lambda item:(-len(item[1]),item[1][0]['sha256'])):
        if len(pins[key])>1:
            excluded.append({'reason':'conflicting_existing_split_near_duplicate_links','count':len(group)});continue
        unique=[]
        for row in sorted(group,key=lambda r:r['sha256']):
            if row['sha256'] not in seen:unique.append(row);seen.add(row['sha256'])
        gid=hashlib.sha256('\n'.join(r['sha256'] for r in unique).encode()).hexdigest()[:20]
        dest=next(iter(pins[key])) if pins[key] else min(parts,key=lambda s:len(parts[s])/ratios[s])
        chosen=rng.sample(unique,min(cap,len(unique))) if dest=='train' else unique
        for row in chosen:row['group']=gid
        parts[dest].extend(chosen);group_info.append({'group':gid,'split':dest,'images':len(chosen),'before_cap':len(unique),'families':sorted({r['source_group'] for r in group}),'pinned_to_existing_split':bool(pins[key])})
    assert all(parts.values()),'independent_development_splits_required'
    out.mkdir(parents=True);manifests.mkdir(parents=True);counts={};manifest_hashes={}
    for split,items in parts.items():
        images=out/split/'images';labels=out/split/'labels';images.mkdir(parents=True);labels.mkdir(parents=True)
        records=[];counts[split]=Counter()
        for row in items:
            original=Path(row['path']);target=images/(row['sha256'][:24]+original.suffix);label=labels/(target.stem+'.txt')
            assert sha256(original)==row['sha256'] and sha256(row['label'])==row['label_sha256']
            shutil.copyfile(original,target)
            converted=[[mapping[int(c)],x,y,w,h] for c,x,y,w,h in row['parsed'] if int(c) in mapping]
            label.write_text('\n'.join(str(c)+' '+' '.join(f'{v:.8f}' for v in (x,y,w,h)) for c,x,y,w,h in converted)+'\n',encoding='utf-8')
            counts[split].update(['head' if c==0 else 'torso' for c,*_ in converted])
            records.append({'source_image':row['path'],'source_label':row['label'],'image_sha256':row['sha256'],'source_label_sha256':row['label_sha256'],
                'image':str(target.relative_to(ROOT)),'label':str(label.relative_to(ROOT)),'derived_label_sha256':sha256(label),'group':row['group'],
                'source_family':row['source_group'],'class_mapping':mapping,'split':split,'transform_version':'body-parts-v1-original-boxes-no-fractional-regions',
                'license':'CC-BY-4.0 per original data.yaml','field_independence_verified':False})
        path=manifests/(split+'.json');write_json(path,{'records':records});manifest_hashes[split]=sha256(path)
        (out/(split+'.txt')).write_text('\n'.join(str((ROOT/r['image']).resolve()) for r in records)+'\n',encoding='utf-8')
    specification=out/'data.yaml';specification.write_text(yaml.safe_dump({'path':str(out),'train':'train.txt','val':'validation.txt','test':'test.txt','names':['head','torso']},allow_unicode=True),encoding='utf-8')
    report={'images':{s:len(v) for s,v in parts.items()},'objects':{s:dict(v) for s,v in counts.items()},'groups':group_info,'excluded':excluded,
        'source_mapping':mapping,'scope':'Anatomical localization only. Hat is not mapped to safety helmet; garment status labels are merged only to locate torso.',
        'semantics_review':'32 negative-containing whole images visually reviewed; existing positive head/vest reviews retained. Not exhaustive label certification.',
        'split_strategy':'Source family/stem/exact hash/dHash<=4 union; preserve prior model split near-duplicate assignments; reject conflicts; train group cap400',
        'user_video_used_for_training':False,'field_independent_test':False,'manifest_sha256':manifest_hashes}
    write_json(manifests/'selection.json',report);print(json.dumps(report['images']),flush=True)
    return specification,manifest_hashes


def train(run_id,epochs):
    output=ROOT/'outputs/training'/run_id
    if output.exists():raise ValueError('training_run_already_exists')
    data,hashes=prepare(run_id)
    from ultralytics import YOLO
    import torch
    initial=ROOT/'models/yolo26n.pt';source=read_json(ROOT/'models/yolo26n.pt.source.json');assert sha256(initial)==source['sha256']
    params=dict(data=str(data),epochs=epochs,imgsz=640,batch=16,device=0,workers=0,seed=51,deterministic=True,optimizer='AdamW',lr0=.001,
        warmup_bias_lr=.001,warmup_epochs=2,freeze=10,nms=False,patience=12,project=str(output.parent),name=run_id,exist_ok=True,amp=True,
        mosaic=.5,mixup=0,scale=.25,translate=.1,close_mosaic=5,plots=True,cache=False,verbose=False)
    output.mkdir(parents=True)
    provenance={'run_id':run_id,'status':'running','smoke':False,'started_at':utc_now(),'initial_weights':initial.name,'initial_weights_sha256':sha256(initial),
        'initial_weights_source':source['url'],'manifest_sha256':hashes,'manifest_paths':{s:f'data/manifests/{run_id}/{s}.json' for s in hashes},
        'parameters':params,'packages':{n:importlib.metadata.version(n) for n in ('torch','torchvision','ultralytics')},'gpu':torch.cuda.get_device_name(0),
        'user_video_used_for_training':False,'field_acceptance':False,'role':'body_part_localizer; not a PPE status classifier'}
    write_json(output/'provenance.json',provenance)
    try:
        YOLO(str(initial)).train(**params);best=output/'weights/best.pt'
        provenance.update(status='completed',completed_at=utc_now(),checkpoint=str(best.relative_to(ROOT)),checkpoint_sha256=sha256(best),checkpoint_selection='best validation fitness')
    except Exception as error:provenance.update(status='failed',error_type=type(error).__name__);raise
    finally:write_json(output/'provenance.json',provenance)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='body-parts-yolo26n-v1');p.add_argument('--epochs',type=int,default=30);a=p.parse_args();train(a.run_id,a.epochs)
