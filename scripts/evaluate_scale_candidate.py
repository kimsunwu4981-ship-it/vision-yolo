"""Select on mixed validation; compare native and synthetic-small test domains."""
import argparse,shutil
from pathlib import Path
import numpy as np,yaml
from app.common import ROOT,read_json,write_json,sha256,utc_now


def native_ppe_spec():
    directory=ROOT/'data/derived/ppe-native-evaluation-two-class';directory.mkdir(exist_ok=True)
    for split in ('validation','test'):
        images=directory/split/'images';labels=directory/split/'labels';images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
        paths=[]
        for r in read_json(ROOT/'data/manifests'/(split+'.json'))['records']:
            source=ROOT/r['image'];target=images/source.name;label=labels/(source.stem+'.txt')
            if not target.exists():shutil.copyfile(source,target)
            converted=[]
            for line in (ROOT/r['label']).read_text().splitlines():
                values=line.split()
                if values[0] not in {'1','2'}:continue
                converted.append(str(int(values[0])-1)+' '+' '.join(values[1:]))
            label.write_text('\n'.join(converted)+'\n',encoding='utf-8');paths.append(str(target))
        (directory/(split+'.txt')).write_text('\n'.join(paths)+'\n',encoding='utf-8')
    path=directory/'data.yaml';path.write_text(yaml.safe_dump(dict(path=str(directory),train='validation.txt',val='validation.txt',test='test.txt',names=['helmet','safety_vest'])),encoding='utf-8')
    return path


def evaluate(run_id,kind,head=None,head_reason=None):
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(1)
    run=ROOT/'outputs/training'/run_id;p=read_json(run/'provenance.json');assert p['status']=='completed'
    checkpoint=ROOT/p['checkpoint'];assert sha256(checkpoint)==p['checkpoint_sha256']
    data=p.get('evaluation_data',p['parameters']['data']);selection_data=p.get('selection_data',data);reports={}
    def one(model_path,spec,split,nms,name,classes=None):
        model=YOLO(str(model_path))
        metric=model.val(data=str(spec),split=split,imgsz=640,batch=12,device=0,workers=0,quantize=16,nms=nms,max_det=600,
            classes=classes,project=str(run/'evaluation'),name=name,plots=False,verbose=False)
        result=dict(classes={model.names[int(c)]:dict(zip(('precision','recall','AP50','AP50_95'),map(float,metric.box.class_result(i)))) for i,c in enumerate(metric.box.ap_class_index)},
            aggregate={k:float(v) for k,v in metric.results_dict.items()},nms=nms,precision_bits=16)
        result['fixed_confidence_025']={model.names[int(c)]:dict(precision=float(np.interp(.25,metric.box.px,metric.box.p_curve[i])),
            recall=float(np.interp(.25,metric.box.px,metric.box.r_curve[i])),iou=.5) for i,c in enumerate(metric.box.ap_class_index)}
        if split=='val':
            curve=np.asarray(metric.box.f1_curve).mean(0);best=int(curve.argmax());result['operating_point']=dict(confidence=float(metric.box.px[best]),macro_f1=float(curve[best]),basis=p.get('validation_basis','mixed validation')+' bbox macro F1; not video/test')
            result['class_operating_points']={model.names[int(c)]:dict(confidence=float(metric.box.px[int(np.asarray(metric.box.f1_curve[i]).argmax())]),
                F1=float(np.asarray(metric.box.f1_curve[i]).max())) for i,c in enumerate(metric.box.ap_class_index)}
        return result
    for nms in (True,False):reports['one_to_many' if nms else 'one_to_one']=one(checkpoint,selection_data,'val',nms,'selection-val-'+str(nms))
    if head and not head_reason:raise ValueError('explicit_head_override_reason_required')
    selected=head or max(reports,key=lambda k:reports[k]['aggregate']['metrics/mAP50-95(B)']);nms=reports[selected]['nms']
    native=read_json(ROOT/'models/person-yolo26s-v5.provenance.json')['parameters']['data'] if kind=='person' else native_ppe_spec()
    result=dict(run_id=run_id,checkpoint_sha256=sha256(checkpoint),evaluated_at=utc_now(),kind=kind,
        validation_heads=reports,selected_head=selected,selected_nms=nms,selected_operating_point=reports[selected]['operating_point'],
        head_selection_reason=head_reason or 'highest validation mAP50-95',diagnostic_video_used_for_head_policy=bool(head),
        mixed_test=one(checkpoint,data,'test',nms,'mixed-test'),native_test=one(checkpoint,native,'test',nms,'native-test'),
        user_video_used_for_training=False,field_acceptance=False,
        limitation='Synthetic composites have correlated source objects. Native development test has been inspected for previous candidates; not independent acceptance.')
    # Existing checkpoint baseline on unchanged native evaluation semantics.
    old=ROOT/'models'/('person-yolo26s-v5.pt' if kind=='person' else 'ppe-yolo26s-v2.pt')
    old_spec=native if kind=='person' else ROOT/'data/derived/ppe_v2/data.yaml'
    result['baseline_native_test']=one(old,old_spec,'test',True,'baseline-native-test',classes=None if kind=='person' else [1,2])
    if kind=='person':result['baseline_mixed_test']=one(old,data,'test',True,'baseline-mixed-test')
    write_json(run/'candidate-evaluation.json',result)
    print({k:result[k] for k in ('run_id','selected_nms','selected_operating_point','native_test','baseline_native_test')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_id');parser.add_argument('--kind',choices=['person','ppe'],required=True)
    parser.add_argument('--head',choices=['one_to_many','one_to_one']);parser.add_argument('--head-reason');args=parser.parse_args();evaluate(args.run_id,args.kind,args.head,args.head_reason)
