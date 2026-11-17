"""Restore native PPE scale weighting without introducing user-video labels."""
import argparse,importlib.metadata
import yaml
from app.common import ROOT,read_json,write_json,sha256,utc_now
from scripts.evaluate_scale_candidate import native_ppe_spec


def main(run_id='ppe-yolo26s-scale-v5'):
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(1)
    run=ROOT/'outputs/training'/run_id
    directory=ROOT/'data/derived'/run_id;manifests=ROOT/'data/manifests'/run_id
    assert not run.exists() and not directory.exists() and not manifests.exists()
    run.mkdir(parents=True);directory.mkdir(parents=True);manifests.mkdir(parents=True)
    previous=ROOT/'data/manifests/ppe-yolo26s-scale-v4';records=[]
    for row in read_json(previous/'train.json')['records']:
        upper=row['source_records'][0]['annotation_schema']=='upper'
        repeats=(5 if upper else 1) if row['style']=='native' else (1 if upper and row['style']=='shrunk' else 0)
        records.extend([{**row,'sampling_repeat':i} for i in range(repeats)])
    native=native_ppe_spec();spec=yaml.safe_load(native.read_text(encoding='utf-8'))
    spec['train']=str(directory/'train.txt');data=directory/'data.yaml'
    data.write_text(yaml.safe_dump(spec),encoding='utf-8')
    (directory/'train.txt').write_text('\n'.join(str(ROOT/r['image']) for r in records)+'\n',encoding='utf-8')
    hashes={}
    for split in ('train','validation','test'):
        rows=records if split=='train' else read_json(ROOT/'data/manifests'/(split+'.json'))['records']
        path=manifests/(split+'.json')
        write_json(path,dict(records=rows,source_manifest_sha256=sha256(previous/'train.json') if split=='train' else sha256(ROOT/'data/manifests'/(split+'.json')),
            evaluation_label_mapping={'1':0,'2':1} if split!='train' else None,user_video_used_for_training=False))
        hashes[split]=sha256(path)
    parent=read_json(ROOT/'outputs/training/ppe-yolo26s-scale-v4/provenance.json');initial=ROOT/parent['checkpoint']
    assert parent['status']=='completed' and sha256(initial)==parent['checkpoint_sha256']
    params=dict(data=str(data),epochs=10,imgsz=640,batch=12,device=0,workers=0,seed=68,deterministic=True,
        optimizer='AdamW',lr0=.00002,warmup_bias_lr=.00002,warmup_epochs=1,weight_decay=.0001,freeze=10,nms=False,
        patience=10,project=str(run.parent),name=run.name,exist_ok=True,cache=False,plots=True,amp=True,
        mosaic=0.,mixup=0.,scale=.1,translate=.05,degrees=10,shear=0,perspective=0,fliplr=.5,close_mosaic=0,max_det=600,verbose=False)
    info=dict(run_id=run_id,status='running',started_at=utc_now(),parameters=params,manifest_sha256=hashes,
        manifest_paths={s:str((manifests/(s+'.json')).relative_to(ROOT)) for s in hashes},
        initial_checkpoint=str(initial.relative_to(ROOT)),initial_checkpoint_sha256=sha256(initial),parent_run_id=parent['run_id'],
        initial_weights_sha256=parent['initial_weights_sha256'],initial_weights_source=parent['initial_weights_source'],
        selection_data=str(data),evaluation_data=str(ROOT/'data/derived/ppe-yolo26s-scale-v4/data.yaml'),validation_basis='unchanged native validation groups',
        training_rows=len(records),strategy='Native upper 5x; existing lower native PPE 1x; upper shrunk 1x; frozen backbone parameters and batch normalization statistics.',
        user_video_used_for_training=False,field_acceptance=False,external_dataset_used=False,
        packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','ultralytics')},gpu=torch.cuda.get_device_name(0))
    model=YOLO(str(initial));frozen={}
    def lock(trainer):
        for i,layer in enumerate(trainer.model.model):
            if i<10:layer.eval()
    def capture(trainer):
        lock(trainer);frozen.update({n:t.detach().cpu().clone() for n,t in trainer.model.state_dict().items() if int(n.split('.')[1])<10})
    def verify(trainer):
        info['frozen_backbone_max_delta']=max(float((t.detach().cpu().float()-frozen[n].float()).abs().max()) for n,t in trainer.model.state_dict().items() if n in frozen)
        assert info['frozen_backbone_max_delta']==0
    model.add_callback('on_pretrain_routine_end',capture);model.add_callback('on_train_batch_start',lock);model.add_callback('on_train_end',verify)
    write_json(run/'provenance.json',info)
    try:
        model.train(**params);best=run/'weights/best.pt'
        info.update(status='completed',completed_at=utc_now(),checkpoint=str(best.relative_to(ROOT)),checkpoint_sha256=sha256(best),checkpoint_selection='best native validation fitness')
    except Exception as error:info.update(status='failed',error_type=type(error).__name__);raise
    finally:write_json(run/'provenance.json',info)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='ppe-yolo26s-scale-v5');main(parser.parse_args().run_id)
