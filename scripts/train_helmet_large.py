"""Train a higher-capacity helmet detector on existing, audited labels only."""
import argparse,importlib.metadata
from app.common import ROOT,read_json,write_json,sha256,utc_now


def main(run_id,epochs):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(1);source=ROOT/'models/yolo26x.pt';origin=read_json(ROOT/'models/yolo26x.pt.source.json')
    data_run='helmet-yolo26s-context-v6';parent=read_json(ROOT/'outputs/training'/data_run/'provenance.json');audit=read_json(ROOT/'data/manifests'/data_run/'audit.json')
    assert all(not any(pair.values()) for pair in audit['pairwise_split_overlap'].values())
    assert sha256(source)=='9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92'
    run=ROOT/'outputs/training'/run_id;assert not run.exists();run.mkdir(parents=True)
    params=dict(data=parent['parameters']['data'],epochs=epochs,imgsz=640,batch=12,device=0,workers=0,seed=79,deterministic=True,optimizer='AdamW',lr0=.0005,
        warmup_bias_lr=.0005,warmup_epochs=1,weight_decay=.0001,freeze=10,nms=False,patience=epochs,project=str(run.parent),name=run.name,exist_ok=True,
        cache=False,plots=True,amp=True,mosaic=0.,mixup=0.,scale=.15,translate=.08,degrees=25,shear=2,perspective=.0002,fliplr=.5,close_mosaic=0,max_det=300,verbose=False)
    p=dict(run_id=run_id,status='running',started_at=utc_now(),parameters=params,manifest_sha256=parent['manifest_sha256'],manifest_paths=parent['manifest_paths'],statistics=parent['statistics'],
        initial_weights_sha256=sha256(source),initial_weights_source='https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt',
        strategy='Official larger initialization; new one-class helmet head; freeze first 10 layers including BN statistics; same audited existing helmet/nohat labels.',
        packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','ultralytics')},gpu=torch.cuda.get_device_name(0),
        user_video_used_for_training=False,external_dataset_used=False,field_acceptance=False)
    model=YOLO(str(source))
    def lock(trainer):
        for i,layer in enumerate(trainer.model.model):
            if i<10:layer.eval()
    model.add_callback('on_train_batch_start',lock);write_json(run/'provenance.json',p)
    try:
        model.train(**params);best=run/'weights/best.pt';p.update(status='completed',completed_at=utc_now(),checkpoint=str(best.relative_to(ROOT)),checkpoint_sha256=sha256(best),checkpoint_selection='existing context validation fitness')
    except Exception as error:p.update(status='failed',error_type=type(error).__name__);raise
    finally:write_json(run/'provenance.json',p)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='helmet-yolo26x-context-v7');parser.add_argument('--epochs',type=int,default=10);a=parser.parse_args();main(a.run_id,a.epochs)
