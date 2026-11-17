"""Evaluate one-class helmet context models without confusing vest label IDs."""
import argparse,shutil
import numpy as np,yaml
from app.common import ROOT,read_json,write_json,sha256,utc_now


def native_spec():
    folder=ROOT/'data/derived/helmet-native-evaluation';folder.mkdir(exist_ok=True)
    for split in ('validation','test'):
        images=folder/split/'images';labels=folder/split/'labels';images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True);paths=[]
        for row in read_json(ROOT/'data/manifests'/(split+'.json'))['records']:
            source=ROOT/row['image'];target=images/source.name
            if not target.exists():shutil.copyfile(source,target)
            objects=[]
            for line in (ROOT/row['label']).read_text().splitlines():
                values=line.split()
                if values and values[0]=='1':objects.append('0 '+' '.join(values[1:]))
            (labels/(target.stem+'.txt')).write_text('\n'.join(objects)+'\n',encoding='utf-8');paths.append(str(target))
        (folder/(split+'.txt')).write_text('\n'.join(paths)+'\n',encoding='utf-8')
    spec=folder/'data.yaml';spec.write_text(yaml.safe_dump(dict(path=str(folder),train='validation.txt',val='validation.txt',test='test.txt',names=['helmet'])),encoding='utf-8');return spec


def main(run_id):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(1);run=ROOT/'outputs/training'/run_id;p=read_json(run/'provenance.json');assert p['status']=='completed'
    checkpoint=ROOT/p['checkpoint'];assert sha256(checkpoint)==p['checkpoint_sha256'];data=p['parameters']['data'];native=native_spec()
    def one(path,spec,split,nms,name,classes=None):
        model=YOLO(str(path));m=model.val(data=str(spec),split=split,imgsz=640,batch=12,device=0,workers=0,quantize=16,nms=nms,
            max_det=300,classes=classes,project=str(run/'evaluation'),name=name,plots=False,verbose=False)
        assert len(m.box.ap_class_index)==1
        i=0;best=int(np.asarray(m.box.f1_curve[i]).argmax())
        return dict(nms=nms,metrics=dict(zip(('precision','recall','AP50','AP50_95'),map(float,m.box.class_result(i)))),
            operating_point=dict(confidence=float(m.box.px[best]),F1=float(m.box.f1_curve[i][best])),
            fixed_thresholds={str(t):dict(precision=float(np.interp(t,m.box.px,m.box.p_curve[i])),recall=float(np.interp(t,m.box.px,m.box.r_curve[i]))) for t in (.25,.5,.7,.8)},
            image_count=int(m.seen) if hasattr(m,'seen') else None)
    reports={key:one(checkpoint,data,'val',nms,'context-val-'+key) for key,nms in [('one_to_many',True),('one_to_one',False)]}
    head=max(reports,key=lambda k:reports[k]['metrics']['AP50_95']);nms=reports[head]['nms']
    result=dict(run_id=run_id,checkpoint_sha256=sha256(checkpoint),evaluated_at=utc_now(),validation_heads=reports,selected_head=head,selected_nms=nms,
        selected_operating_point=reports[head]['operating_point'],context_test=one(checkpoint,data,'test',nms,'context-test'),
        native_test=one(checkpoint,native,'test',nms,'native-test'),
        baseline_native_test=one(ROOT/'models/ppe-yolo26s-v2.pt',ROOT/'data/derived/ppe_v2/data.yaml','test',True,'baseline-native',classes=[1]),
        user_video_used_for_training=False,field_acceptance=False,
        limitation='Existing development splits; context crops share source images. Not independent field recall. Threshold still needs bare-head false-positive checks.')
    write_json(run/'candidate-evaluation.json',result);print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_id');main(parser.parse_args().run_id)
