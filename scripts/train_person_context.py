"""Adapt official person classification rows without discarding pretrained features."""
import argparse,importlib.metadata,copy
from app.common import ROOT,read_json,write_json,sha256,utc_now


def train(run_id,epochs):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(1)
    source=ROOT/'models/yolo26x.pt';source_info=read_json(source.with_suffix('.pt.source.json'))
    assert sha256(source)==source_info['sha256']=='9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92'
    run=ROOT/'outputs/training'/run_id;assert not run.exists();run.mkdir(parents=True)
    model=YOLO(str(source));head=model.model.model[-1];last=len(model.model.model)-1
    for branch in ('cv3','one2one_cv3'):
        for scale in range(3):
            old=getattr(head,branch)[scale][-1];new=torch.nn.Conv2d(old.in_channels,1,old.kernel_size,old.stride,old.padding,bias=True)
            new.weight.data.copy_(old.weight.data[:1]);new.bias.data.copy_(old.bias.data[:1]);getattr(head,branch)[scale][-1]=new
    head.nc=1;head.no=head.reg_max*4+1;model.model.names={0:'person'};model.model.yaml['nc']=1
    init=run/'person-initialization.pt';model.save(str(init));model=YOLO(str(init))
    freeze=list(range(last))+[f'{last}.cv2',f'{last}.one2one_cv2']
    for branch in ('cv3','one2one_cv3'):
        freeze += [f'{last}.{branch}.{scale}.{block}' for scale in range(3) for block in (0,1)]
    data=ROOT/'data/derived/person-yolo26s-scale-v7/data.yaml'
    manifest_parent=ROOT/'data/manifests/person-yolo26s-scale-v7'
    params=dict(data=str(data),epochs=epochs,imgsz=640,batch=8,device=0,workers=0,seed=71,deterministic=True,optimizer='AdamW',
        lr0=.00001,warmup_bias_lr=.00001,warmup_epochs=1,weight_decay=.0001,freeze=freeze,nms=False,patience=epochs,
        project=str(run.parent),name=run.name,exist_ok=True,cache=False,plots=True,amp=True,
        mosaic=0.,mixup=0.,scale=.05,translate=.03,degrees=5,shear=0,perspective=0,fliplr=.5,close_mosaic=0,max_det=600,verbose=False)
    p=dict(run_id=run_id,status='running',started_at=utc_now(),parameters=params,
        initial_weights_sha256=sha256(source),initial_weights_source=source_info['url'],initial_person_slice_sha256=sha256(init),
        manifest_paths={s:str((manifest_parent/(s+'.json')).relative_to(ROOT)) for s in ('train','validation','test')},
        manifest_sha256={s:sha256(manifest_parent/(s+'.json')) for s in ('train','validation','test')},
        selection_data=str(data),evaluation_data=str(data),validation_basis='unchanged native validation groups',
        strategy='Preserve official class-0 projections when reducing nc=80 to nc=1; train only six native person classification projections; freeze all other weights and batch normalization statistics.',
        packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','ultralytics')},gpu=torch.cuda.get_device_name(0),
        user_video_used_for_training=False,external_dataset_used=False,field_acceptance=False)
    fixed={};learned={}
    def lock(trainer):
        for module in trainer.model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
    def capture(trainer):
        lock(trainer);trainable={n for n,x in trainer.model.named_parameters() if x.requires_grad}
        p['trainable_parameters']={n:x.numel() for n,x in trainer.model.named_parameters() if x.requires_grad}
        for n,x in trainer.model.state_dict().items():(learned if n in trainable else fixed)[n]=x.detach().cpu().clone()
        write_json(run/'provenance.json',p)
    def verify(trainer):
        state=trainer.model.state_dict()
        p['frozen_weights_and_buffers_max_delta']=max(float((state[n].detach().cpu().float()-x.float()).abs().max()) for n,x in fixed.items())
        p['trainable_weight_l2_change']={n:float((state[n].detach().cpu()-x).norm()) for n,x in learned.items()}
        assert p['frozen_weights_and_buffers_max_delta']==0 and any(p['trainable_weight_l2_change'].values())
    model.add_callback('on_pretrain_routine_end',capture);model.add_callback('on_train_batch_start',lock);model.add_callback('on_train_end',verify)
    write_json(run/'provenance.json',p)
    try:
        model.train(**params);best=run/'weights/best.pt'
        p.update(status='completed',completed_at=utc_now(),checkpoint=str(best.relative_to(ROOT)),checkpoint_sha256=sha256(best),checkpoint_selection='best native validation fitness')
    except Exception as error:p.update(status='failed',error_type=type(error).__name__);raise
    finally:write_json(run/'provenance.json',p)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='person-yolo26x-context-v8');parser.add_argument('--epochs',type=int,default=6);a=parser.parse_args();train(a.run_id,a.epochs)
