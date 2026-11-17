"""Retain native scenes and frozen backbone statistics during scale adaptation."""
import argparse, random, importlib.metadata
import cv2, numpy as np, yaml
from app.common import ROOT, read_json, write_json, sha256, utc_now


def main(run_id='person-yolo26s-scale-v7'):
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(1)
    run = ROOT/'outputs/training'/run_id
    directory = ROOT/'data/derived'/run_id
    manifests = ROOT/'data/manifests'/run_id
    assert not run.exists() and not directory.exists() and not manifests.exists()
    run.mkdir(parents=True); directory.mkdir(parents=True); manifests.mkdir(parents=True)
    parent = ROOT/'data/manifests/person-yolo26s-v5'
    original = read_json(parent/'train.json')['records']
    scaled = read_json(ROOT/'data/manifests/person-yolo26s-scale-v6/train.json')['records']
    records = []
    for record in original:
        rows = [list(map(float, line.split())) for line in (ROOT/record['label']).read_text().splitlines() if line.strip()]
        # Repetition changes sampling weight, not independent sample count.
        small = any(row[4]*640 < 160 for row in rows)
        records.extend([{**record, 'sampling_repeat': i, 'user_video_used': False} for i in range(9 if small else 3)])
    records += [r for r in scaled if r['style']=='shrunk']
    rng = random.Random(67)
    images = directory/'train/images'; labels = directory/'train/labels'
    images.mkdir(parents=True); labels.mkdir(parents=True)
    negatives = 0
    for record in original:
        boxes = []
        for line in (ROOT/record['label']).read_text().splitlines():
            if not line.strip(): continue
            _,x,y,w,h = map(float,line.split()); boxes.append(((x-w/2)*640,(y-h/2)*640,(x+w/2)*640,(y+h/2)*640))
        for attempt in range(20):
            edge = rng.choice([192,256,320]); left = rng.randrange(641-edge); top = rng.randrange(641-edge)
            # Preserve a 16 px exclusion margin around every annotated person.
            if any(left < x2+16 and left+edge > x1-16 and top < y2+16 and top+edge > y1-16 for x1,y1,x2,y2 in boxes): continue
            source = cv2.imdecode(np.fromfile(ROOT/record['image'],np.uint8),cv2.IMREAD_COLOR)
            crop = cv2.resize(source[top:top+edge,left:left+edge],(640,640))
            image = images/f'background-{negatives:04d}.jpg'; label = labels/(image.stem+'.txt')
            ok,encoded = cv2.imencode('.jpg',crop); assert ok
            image.write_bytes(encoded.tobytes()); label.write_text('',encoding='utf-8')
            records.append(dict(image=str(image.relative_to(ROOT)),label=str(label.relative_to(ROOT)),image_sha256=sha256(image),derived_label_sha256=sha256(label),
                split='train',source_records=[record],transform=dict(crop_xyxy=[left,top,left+edge,top+edge],margin=16),style='annotated-background',user_video_used=False,
                limitation='Negative region relative to existing annotations; annotation completeness is inherited.'))
            negatives += 1; break
        if negatives >= 200: break
    hashes = {}
    for split in ('train','validation','test'):
        selected = records if split=='train' else read_json(parent/(split+'.json'))['records']
        path = manifests/(split+'.json')
        write_json(path,dict(records=selected,source_manifest_sha256=sha256(parent/(split+'.json')),user_video_used_for_training=False))
        hashes[split] = sha256(path)
        (directory/(split+'.txt')).write_text('\n'.join(str(ROOT/r['image']) for r in selected)+'\n',encoding='utf-8')
    specification = directory/'data.yaml'
    specification.write_text(yaml.safe_dump(dict(path=str(directory),train='train.txt',val='validation.txt',test='test.txt',names=['person'])),encoding='utf-8')
    initial = ROOT/'models/person-yolo26s-v5.pt'; parent_info = read_json(ROOT/'models/person-yolo26s-v5.provenance.json')
    assert sha256(initial)==parent_info['checkpoint_sha256']
    params = dict(data=str(specification),epochs=10,imgsz=640,batch=12,device=0,workers=0,seed=67,deterministic=True,
        optimizer='AdamW',lr0=.00002,warmup_bias_lr=.00002,warmup_epochs=1,weight_decay=.0001,freeze=10,nms=False,
        patience=10,project=str(run.parent),name=run.name,exist_ok=True,cache=False,plots=True,amp=True,
        mosaic=0.,mixup=0.,scale=.1,translate=.05,degrees=10,shear=0,perspective=0,fliplr=.5,close_mosaic=0,max_det=600,verbose=False)
    info = dict(run_id=run_id,status='running',started_at=utc_now(),parameters=params,manifest_sha256=hashes,
        manifest_paths={s:str((manifests/(s+'.json')).relative_to(ROOT)) for s in hashes},
        initial_checkpoint=str(initial.relative_to(ROOT)),initial_checkpoint_sha256=sha256(initial),parent_run_id=parent_info['run_id'],
        initial_weights_sha256=parent_info['initial_weights_sha256'],initial_weights_source=parent_info['initial_weights_source'],
        selection_data=str(specification),evaluation_data=str(ROOT/'data/derived/person-yolo26s-scale-v6/data.yaml'),
        validation_basis='unchanged native validation groups',training_rows=len(records),native_source_images=len(original),background_crops=negatives,
        strategy='Native 3x; existing naturally small images 9x; shrunk scene 1x; annotated background crops; frozen backbone parameters and batch normalization statistics.',
        user_video_used_for_training=False,field_acceptance=False,external_dataset_used=False,
        packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','ultralytics')},gpu=torch.cuda.get_device_name(0))
    model=YOLO(str(initial)); frozen_start={}
    def lock_backbone(trainer):
        for index, layer in enumerate(trainer.model.model):
            if index < 10: layer.eval()
    def capture(trainer):
        lock_backbone(trainer)
        frozen_start.update({n:t.detach().cpu().clone() for n,t in trainer.model.state_dict().items() if int(n.split('.')[1])<10})
    def verify(trainer):
        info['frozen_backbone_max_delta']=max(float((t.detach().cpu().float()-frozen_start[n].float()).abs().max()) for n,t in trainer.model.state_dict().items() if n in frozen_start)
        assert info['frozen_backbone_max_delta']==0
    model.add_callback('on_pretrain_routine_end',capture)
    model.add_callback('on_train_batch_start',lock_backbone)
    model.add_callback('on_train_end',verify)
    write_json(run/'provenance.json',info)
    try:
        model.train(**params); best=run/'weights/best.pt'
        info.update(status='completed',completed_at=utc_now(),checkpoint=str(best.relative_to(ROOT)),checkpoint_sha256=sha256(best),checkpoint_selection='best native validation fitness')
    except Exception as error:
        info.update(status='failed',error_type=type(error).__name__); raise
    finally: write_json(run/'provenance.json',info)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='person-yolo26s-scale-v7');main(parser.parse_args().run_id)
