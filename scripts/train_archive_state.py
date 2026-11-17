"""Train positive/explicit-negative PPE state boxes from existing local archives."""
import argparse,importlib.metadata
from app.common import ROOT,read_json,write_json,sha256,utc_now


def train(kind,version,epochs):
    import torch,yaml
    from ultralytics import YOLO
    torch.set_num_threads(1);run_id=kind+'-state-yolo26s-'+version;run=ROOT/'outputs/training'/run_id;assert not run.exists();run.mkdir(parents=True)
    data=ROOT/'data/derived'/run_id/'data.yaml';spec=yaml.safe_load(data.read_text(encoding='utf-8'));names=spec['names']
    source=ROOT/'models/ppe-yolo26s-v2.pt';parent=read_json(ROOT/'models/ppe-yolo26s-v2.provenance.json');assert sha256(source)==parent['checkpoint_sha256']
    manifests={s:ROOT/'data/manifests'/run_id/(s+'.json') for s in ('train','validation','test')}
    model=YOLO(str(source));head=model.model.model[-1];positive_row=1 if kind=='head' else 2
    for branch in ('cv3','one2one_cv3'):
        for scale in range(3):
            old=getattr(head,branch)[scale][-1];new=torch.nn.Conv2d(old.in_channels,2,old.kernel_size,old.stride,old.padding,bias=True)
            new.weight.data[0].copy_(old.weight.data[positive_row]);new.bias.data[0].copy_(old.bias.data[positive_row])
            # New explicit-negative class learns from its existing annotations.
            torch.nn.init.normal_(new.weight.data[1],0,.01);new.bias.data[1]=-4.
            getattr(head,branch)[scale][-1]=new
    head.nc=2;head.no=head.reg_max*4+2;model.model.names=dict(enumerate(names));model.model.yaml['nc']=2
    initial=run/'state-initialization.pt';model.save(str(initial));model=YOLO(str(initial))
    params=dict(data=str(data),epochs=epochs,imgsz=640,batch=24,device=0,workers=0,seed=83,deterministic=True,optimizer='AdamW',lr0=.0002,
        warmup_bias_lr=.0002,warmup_epochs=1,weight_decay=.0001,freeze=6,nms=False,patience=epochs,project=str(run.parent),name=run.name,exist_ok=True,
        cache=False,plots=True,amp=True,mosaic=.1,mixup=0.,scale=.25,translate=.08,degrees=15,fliplr=.5,close_mosaic=3,max_det=300,verbose=False)
    p=dict(run_id=run_id,status='running',started_at=utc_now(),parameters=params,manifest_sha256={s:sha256(path) for s,path in manifests.items()},manifest_paths={s:str(path.relative_to(ROOT)) for s,path in manifests.items()},
        initial_weights_sha256=parent['initial_weights_sha256'],initial_weights_source=parent['initial_weights_source'],parent_checkpoint_sha256=sha256(source),parent_run_id=parent['run_id'],
        strategy='Retain project PPE positive class row; learn explicit existing bare-head/bare-torso labels. Native archive images; early feature layers and BN statistics frozen.',
        class_names=names,packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','ultralytics')},gpu=torch.cuda.get_device_name(0),
        user_video_used_for_training=False,new_external_download=False,field_acceptance=False)
    def lock(trainer):
        for i,layer in enumerate(trainer.model.model):
            if i<6:layer.eval()
    model.add_callback('on_train_batch_start',lock);write_json(run/'provenance.json',p)
    try:
        model.train(**params);best=run/'weights/best.pt';p.update(status='completed',completed_at=utc_now(),checkpoint=str(best.relative_to(ROOT)),checkpoint_sha256=sha256(best),checkpoint_selection='existing archive development validation fitness')
    except Exception as error:p.update(status='failed',error_type=type(error).__name__);raise
    finally:write_json(run/'provenance.json',p)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['head','vest'],required=True);parser.add_argument('--version',default='v8');parser.add_argument('--epochs',type=int,default=12);a=parser.parse_args();train(a.kind,a.version,a.epochs)
