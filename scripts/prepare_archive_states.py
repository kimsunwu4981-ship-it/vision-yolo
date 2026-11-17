"""Import existing local COCO archives; no network, legacy code, or video training."""
import argparse,csv,hashlib,json,zipfile,re
from pathlib import Path
from io import BytesIO
from collections import defaultdict,Counter
import av,numpy as np,yaml
from PIL import Image
from app.common import ROOT,read_json,write_json,sha256,utc_now
from app.orientation import decode_bgr
from scripts.prepare_data import UnionFind


def dhash(image):
    gray=np.asarray(image.convert('L').resize((9,8)),dtype=np.int16)
    return int(''.join(map(str,(gray[:,1:]>gray[:,:-1]).astype(int).flatten())),2)


def prepare(version='v8'):
    folder=ROOT/'data/manifests'/('archive-states-'+version);assert not folder.exists();folder.mkdir(parents=True)
    source_root=Path('D:/vigent_private_data/public_ppe/roboflow')
    specs={'head':('hard_hat_workers',{2:0,1:1},['helmet','bare_head']),
           'vest':('safety_vests',{2:0,1:1},['safety_vest','bare_torso'])}
    inventory=list(csv.DictReader((ROOT/'data/manifests/dataset_inventory.csv').open(encoding='utf-8-sig')))
    old_hashes=np.array(sorted({int(r['dhash'],16) for r in inventory if r.get('dhash')}),dtype=np.uint64)
    old_sha={r['sha256'] for r in inventory if r.get('sha256')};video_hashes=[];video_sources=[]
    # User videos are exclusion references only: no pixels, crops or inferred
    # labels from them are ever placed into a training/validation dataset.
    for name in ('크레인 재해.mp4','크레인 점검.mp4'):
        path=ROOT.parent.parent/name;video_sources.append(dict(path=str(path),sha256=sha256(path)))
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                im,_=decode_bgr(frame);video_hashes.append(dhash(Image.fromarray(im[:,:,::-1])))
    video_hashes=np.array(sorted(set(video_hashes)),dtype=np.uint64);pop=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)
    def distances(values,value):return pop[(values^np.uint64(value)).view(np.uint8).reshape(-1,8)].sum(axis=1)
    rows=[];excluded=Counter();archives={};annotation_bytes={};archive_handles={}
    try:
        for kind,(name,mapping,names) in specs.items():
            path=source_root/(name+'.zip');z=zipfile.ZipFile(path);archive_handles[kind]=z
            docs={n:z.read(n).decode('utf-8') for n in z.namelist() if n.startswith('README')};assert 'License:' in docs['README.dataset.txt']
            archives[kind]=dict(path=str(path),sha256=sha256(path),bytes=path.stat().st_size,modified_before_task=path.stat().st_mtime,documentation=docs,class_mapping=mapping,names=names)
            for member in sorted(n for n in z.namelist() if n.endswith('_annotations.coco.json')):
                raw_annotations=z.read(member);annotation_bytes[(kind,member)]=raw_annotations;data=json.loads(raw_annotations);by_image=defaultdict(list)
                categories={c['id']:c['name'] for c in data['categories']}
                expected={2:'helmet',1:'head'} if kind=='head' else {2:'vest',1:'no-vest'}
                assert all(categories[k]==v for k,v in expected.items())
                for ann in data['annotations']:by_image[ann['image_id']].append(ann)
                split={'train':'train','valid':'validation','test':'test'}[member.split('/')[0]]
                for item in data['images']:
                    filename=member.split('/')[0]+'/'+item['file_name'];anns=by_image[item['id']]
                    # Existing HHW person-only labels do not certify head status.
                    if kind=='head' and any(a['category_id']==3 for a in anns):excluded['incomplete_head_status_person_only']+=1;continue
                    stem=Path(item['file_name']).stem.split('.rf.')[0]
                    if kind=='head' and stem.startswith(('003215_','001839_')):excluded['reviewed_incorrect_helmet_on_bare_head']+=1;continue
                    payload=z.read(filename);digest=hashlib.sha256(payload).hexdigest()
                    im=Image.open(BytesIO(payload)).convert('RGB');im.load();w,h=im.size;assert (w,h)==(item['width'],item['height'])
                    fingerprint=dhash(im)
                    if digest in old_sha or np.any(distances(old_hashes,fingerprint)<=4):excluded['existing_dataset_exact_or_near_duplicate']+=1;continue
                    if np.any(distances(video_hashes,fingerprint)<=8):excluded['user_video_near_duplicate']+=1;continue
                    objects=[];invalid=False
                    for ann in anns:
                        if ann['category_id'] not in mapping:continue
                        x,y,bw,bh=map(float,ann['bbox'])
                        if not np.isfinite([x,y,bw,bh]).all() or bw<=0 or bh<=0 or x<-.5 or y<-.5 or x+bw>w+.5 or y+bh>h+.5:invalid=True;break
                        x1,y1=max(0,x),max(0,y);x2,y2=min(w,x+bw),min(h,y+bh)
                        objects.append([mapping[ann['category_id']],(x1+x2)/2/w,(y1+y2)/2/h,(x2-x1)/w,(y2-y1)/h])
                    if invalid:excluded['invalid_coco_box']+=1;continue
                    if not objects:excluded['no_reviewed_status_annotations']+=1;continue
                    rows.append(dict(kind=kind,archive_member=filename,annotation_member=member,annotation_image_id=item['id'],image_sha256=digest,dhash=f'{fingerprint:016x}',
                        width=w,height=h,objects=objects,original_split=split,stem=stem))
            print(kind,'decoded',sum(r['kind']==kind for r in rows),'candidates',flush=True)
        uf=UnionFind(len(rows));owners={};hashes=np.array([int(r['dhash'],16) for r in rows],dtype=np.uint64)
        for i,row in enumerate(rows):
            for tag,key in [('hash',row['image_sha256']),('stem',(row['kind'],row['stem']))]:
                if (tag,key) in owners:uf.union(i,owners[(tag,key)])
                else:owners[(tag,key)]=i
            for offset in np.flatnonzero(distances(hashes[i+1:],hashes[i])<=4):uf.union(i,i+1+int(offset))
        groups=defaultdict(list)
        for i,row in enumerate(rows):groups[uf.find(i)].append(row)
        kept=[];seen=set();group_rows=[]
        for members in groups.values():
            splits={r['original_split'] for r in members}
            if len(splits)>1:excluded['cross_archive_or_near_duplicate_split_conflict']+=len(members);continue
            gid=hashlib.sha256('\n'.join(sorted(r['image_sha256'] for r in members)).encode()).hexdigest()[:24]
            for row in members:
                identity=(row['kind'],row['image_sha256'])
                if identity in seen:excluded['within_split_duplicate']+=1;continue
                seen.add(identity);row['group']=gid;kept.append(row)
            group_rows.append(dict(group=gid,split=next(iter(splits)),members=len(members)))
        statistics={};all_records={}
        for kind,(_,mapping,names) in specs.items():
            run_id=kind+'-state-yolo26s-'+version;derived=ROOT/'data/derived'/run_id;manifest=ROOT/'data/manifests'/run_id
            assert not derived.exists() and not manifest.exists();derived.mkdir(parents=True);manifest.mkdir(parents=True);annotation_paths={}
            for (k,member),payload in annotation_bytes.items():
                if k!=kind:continue
                path=derived/(member.split('/')[0]+'-source-coco.json');path.write_bytes(payload);annotation_paths[member]=path
            statistics[kind]={};all_records[kind]={}
            for split in ('train','validation','test'):
                images=derived/split/'images';labels=derived/split/'labels';images.mkdir(parents=True);labels.mkdir(parents=True);records=[];counts=Counter()
                for row in sorted((r for r in kept if r['kind']==kind and r['original_split']==split),key=lambda r:r['image_sha256']):
                    image=images/(row['image_sha256'][:24]+'.jpg');label=labels/(image.stem+'.txt');payload=archive_handles[kind].read(row['archive_member']);image.write_bytes(payload)
                    assert sha256(image)==row['image_sha256']
                    label.write_text(''.join(str(c)+' '+' '.join(f'{v:.8f}' for v in (x,y,w,h))+'\n' for c,x,y,w,h in row['objects']),encoding='utf-8');counts.update(names[int(o[0])] for o in row['objects'])
                    source_annotation=annotation_paths[row['annotation_member']]
                    records.append({**row,'image':str(image.relative_to(ROOT)),'label':str(label.relative_to(ROOT)),'derived_label_sha256':sha256(label),'split':split,
                        'source_archive':archives[kind]['path'],'source_archive_sha256':archives[kind]['sha256'],
                        'source_records':[dict(image=str(image.relative_to(ROOT)),label=str(source_annotation.relative_to(ROOT)),image_sha256=row['image_sha256'],derived_label_sha256=sha256(source_annotation),group=row['group'])],
                        'user_video_used':False,'transform':'Existing COCO rectangles to normalized YOLO; native pixels unchanged; head status is not person supervision'})
                assert records,'empty_development_split'
                write_json(manifest/(split+'.json'),dict(records=records));(derived/(split+'.txt')).write_text('\n'.join(str(ROOT/r['image']) for r in records)+'\n',encoding='utf-8')
                statistics[kind][split]=dict(images=len(records),objects=dict(counts));all_records[kind][split]=records
            (derived/'data.yaml').write_text(yaml.safe_dump(dict(path=str(derived),train='train.txt',val='validation.txt',test='test.txt',names=names)),encoding='utf-8')
            write_json(manifest/'source.json',archives[kind])
        write_json(folder/'selection.json',dict(created_at=utc_now(),archives=archives,statistics=statistics,excluded=dict(excluded),groups=group_rows,
            existing_dataset_inventory_sha256=sha256(ROOT/'data/manifests/dataset_inventory.csv'),user_video_exclusion_references=video_sources,
            user_video_used_for_training=False,new_external_download=False,
            split_strategy='Keep provided split, exclude cross-split exact/stem/dHash<=4 components across both archives; exclude all existing dataset near duplicates; user video dHash<=8 exclusion only.',
            limitation='No source-site/video IDs for many archive images. Visual review is sampled; source annotation mistakes remain possible. This is development evaluation, not independent field acceptance.'))
        print(statistics,excluded,flush=True)
    finally:
        for z in archive_handles.values():z.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',default='v8');prepare(p.parse_args().version)
