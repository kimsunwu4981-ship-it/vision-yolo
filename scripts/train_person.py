"""Person-only classifier adaptation with pretrained localization preserved."""
import argparse,shutil,json,importlib.metadata
from pathlib import Path
import yaml
from app.common import ROOT,read_json,write_json,sha256,utc_now


def train(run_id,epochs=8,train_box_heads=False,dataset_run=None):
    output=ROOT/'outputs/training'/run_id
    if output.exists():raise ValueError('run_already_exists')
    from ultralytics import YOLO
    import torch
    parent=ROOT/'data/manifests/ppe-yolo26s-balanced-v3'
    data=ROOT/'data/derived'/(dataset_run or run_id);manifests=ROOT/'data/manifests'/run_id
    data.mkdir(parents=True,exist_ok=True);manifests.mkdir(parents=True,exist_ok=True)
    hashes={}
    for split in ('train','validation','test'):
        records=read_json(parent/(split+'.json'))['records'];new=[]
        images=data/split/'images';labels=data/split/'labels';images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
        for record in records:
            original=ROOT/record['image'];image=images/original.name;label=labels/(original.stem+'.txt')
            if not image.exists():shutil.copyfile(original,image)
            lines=[line for line in (ROOT/record['label']).read_text().splitlines() if line.split()[0]=='0']
            label.write_text('\n'.join(lines)+'\n',encoding='utf-8')
            new.append({**record,'image':str(image.relative_to(ROOT)),'label':str(label.relative_to(ROOT)),'derived_label_sha256':sha256(label),'transform_version':'person-only; original full-person labels unchanged','active_classes':['person']})
        path=manifests/(split+'.json');write_json(path,{'records':new,'source_manifest_sha256':sha256(parent/(split+'.json'))});hashes[split]=sha256(path)
        (data/(split+'.txt')).write_text('\n'.join(str((ROOT/r['image']).resolve()) for r in new)+'\n',encoding='utf-8')
    specification=data/'data.yaml';specification.write_text(yaml.safe_dump({'path':str(data),'train':'train.txt','val':'validation.txt','test':'test.txt','names':['person']},allow_unicode=True),encoding='utf-8')
    initial=ROOT/'models/yolo26s.pt';assert sha256(initial)=='646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b'
    model=YOLO(str(initial));last=len(model.model.model)-1
    freeze=list(range(last))
    if not train_box_heads:freeze += [f'{last}.cv2',f'{last}.one2one_cv2']
    for branch in ('cv3','one2one_cv3'):
        for scale in range(3):
            for block in (0,1):freeze.append(f'{last}.{branch}.{scale}.{block}')
    output.mkdir(parents=True)
    params=dict(data=str(specification),epochs=epochs,imgsz=640,batch=16,device=0,workers=0,seed=44,deterministic=True,
        optimizer='AdamW',lr0=.0001,warmup_bias_lr=.0001,warmup_epochs=1,weight_decay=.0001,freeze=freeze,
        nms=False,patience=epochs,project=str(output.parent),name=output.name,exist_ok=True,cache=False,plots=True,amp=True,
        mosaic=0.,mixup=0.,scale=.1,translate=.05,close_mosaic=0,verbose=False)
    provenance={'run_id':run_id,'status':'running','smoke':False,'started_at':utc_now(),'initial_weights':'yolo26s.pt','initial_weights_sha256':sha256(initial),
        'initial_weights_source':'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt','manifest_sha256':hashes,
        'manifest_paths':{s:str((manifests/(s+'.json')).relative_to(ROOT)) for s in hashes},'parameters':params,
        'training_strategy':('Train person box heads and classification output projections; preserve backbone, neck and classification features.' if train_box_heads else 'Train both native person classification output projections; freeze backbone, neck, box regression and classification features to prevent PPE-scene localization forgetting.'),
        'packages':{n:importlib.metadata.version(n) for n in ('torch','torchvision','ultralytics')},'gpu':torch.cuda.get_device_name(0),
        'user_video_used_for_training':False,'field_acceptance':False}
    baseline={}
    def capture_start(trainer):
        baseline.update({n:p.detach().cpu().clone() for n,p in trainer.model.named_parameters() if p.requires_grad})
        provenance['trainable_parameters']={n:p.numel() for n,p in trainer.model.named_parameters() if p.requires_grad}
        write_json(output/'provenance.json',provenance)
    def capture_end(trainer):
        provenance['final_trainable_weight_l2_change']={n:float((p.detach().cpu()-baseline[n]).norm()) for n,p in trainer.model.named_parameters() if n in baseline}
    model.add_callback('on_pretrain_routine_end',capture_start);model.add_callback('on_train_end',capture_end)
    write_json(output/'provenance.json',provenance)
    try:
        model.train(**params)
        best=output/'weights/best.pt'
        assert any(v>0 for v in provenance['final_trainable_weight_l2_change'].values())
        provenance.update(status='completed',completed_at=utc_now(),checkpoint=str(best.relative_to(ROOT)),checkpoint_sha256=sha256(best),checkpoint_selection='highest original validation fitness with one-to-one head')
    except Exception as error:provenance.update(status='failed',error_type=type(error).__name__);raise
    finally:write_json(output/'provenance.json',provenance)
    print(json.dumps({'run_id':run_id,'checkpoint_sha256':sha256(best)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='person-yolo26s-v4');p.add_argument('--epochs',type=int,default=8);p.add_argument('--train-box-heads',action='store_true');p.add_argument('--dataset-run');a=p.parse_args();train(a.run_id,a.epochs,a.train_box_heads,a.dataset_run)
