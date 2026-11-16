"""Select native YOLO26 head on validation, then report the development test split."""
import argparse,json
from app.common import ROOT,read_json,write_json,sha256,utc_now


def evaluate(run_id):
    from ultralytics import YOLO
    run=ROOT/'outputs/training'/run_id;provenance=read_json(run/'provenance.json')
    if provenance.get('status')!='completed':raise ValueError('completed_training_required')
    checkpoint=ROOT/provenance['checkpoint']
    assert sha256(checkpoint)==provenance['checkpoint_sha256']
    data=provenance['parameters']['data'];reports={}
    def one(split,nms,name):
        model=YOLO(str(checkpoint))
        metric=model.val(data=data,split=split,imgsz=640,batch=16,device=0,workers=0,quantize=16,nms=nms,
            project=str(run/'evaluation'),name=name,plots=True,verbose=False)
        classes={}
        for index,cls in enumerate(metric.box.ap_class_index):
            precision,recall,ap50,ap=metric.box.class_result(index)
            classes[model.names[int(cls)]]={'precision':float(precision),'recall':float(recall),'AP50':float(ap50),'AP50_95':float(ap)}
        result={'classes':classes,'aggregate':{k:float(v) for k,v in metric.results_dict.items()},'nms':nms,'precision_bits':16}
        if split=='val':
            import numpy as np
            f1=np.asarray(metric.box.f1_curve).mean(axis=0);best=int(f1.argmax())
            result['validation_operating_point']={'confidence':float(metric.box.px[best]),'macro_f1':float(f1[best]),'selection':'validation bbox macro F1 only'}
        return result
    for nms in (False,True):
        name='one_to_many' if nms else 'one_to_one'
        reports[name]=one('val',nms,'validation_'+name)
    selected=max(reports,key=lambda key:reports[key]['aggregate']['metrics/mAP50-95(B)'])
    nms=reports[selected]['nms']
    result={'run_id':run_id,'checkpoint_sha256':sha256(checkpoint),'evaluated_at':utc_now(),'validation_heads':reports,
        'selected_head':selected,'selected_nms':nms,'head_selection':'highest original validation mAP50-95; not diagnostic video or test',
        'test':one('test',nms,'test_selected_head'),'field_independent_test':False,'user_video_used_for_training':False,'deployment_acceptance':False,
        'test_scope':'Existing development image split, not a fresh blind acceptance set; multiple candidates are reported.'}
    write_json(run/'candidate-evaluation.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_id');evaluate(p.parse_args().run_id)
