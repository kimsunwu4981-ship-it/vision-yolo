"""Select the detection head using existing archive validation, then report test."""
import argparse
import numpy as np
from app.common import ROOT,read_json,write_json,sha256,utc_now


def main(run_id):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(1);run=ROOT/'outputs/training'/run_id;p=read_json(run/'provenance.json');assert p['status']=='completed'
    checkpoint=ROOT/p['checkpoint'];assert sha256(checkpoint)==p['checkpoint_sha256']
    def one(split,nms,name):
        model=YOLO(str(checkpoint));metric=model.val(data=p['parameters']['data'],split=split,imgsz=640,batch=24,device=0,workers=0,quantize=16,nms=nms,
            max_det=300,project=str(run/'evaluation'),name=name,plots=False,verbose=False)
        classes={}
        for i,c in enumerate(metric.box.ap_class_index):
            best=int(np.asarray(metric.box.f1_curve[i]).argmax());classes[model.names[int(c)]]=dict(zip(('precision','recall','AP50','AP50_95'),map(float,metric.box.class_result(i))))
            classes[model.names[int(c)]].update(best_F1_threshold=float(metric.box.px[best]),fixed_thresholds={str(t):dict(precision=float(np.interp(t,metric.box.px,metric.box.p_curve[i])),recall=float(np.interp(t,metric.box.px,metric.box.r_curve[i]))) for t in (.25,.5,.7,.85)})
        return dict(nms=nms,classes=classes,aggregate={k:float(v) for k,v in metric.results_dict.items()})
    heads={key:one('val',nms,'val-'+key) for key,nms in [('one_to_many',True),('one_to_one',False)]};selected=max(heads,key=lambda k:heads[k]['aggregate']['metrics/mAP50-95(B)'])
    result=dict(run_id=run_id,checkpoint_sha256=sha256(checkpoint),evaluated_at=utc_now(),validation_heads=heads,selected_head=selected,selected_nms=heads[selected]['nms'],
        test=one('test',heads[selected]['nms'],'test-selected'),user_video_used_for_training=False,field_acceptance=False,
        limitation='Existing archive development split, sampled annotation review and hash-based grouping; source-site metadata incomplete. Not independent CCTV acceptance.')
    write_json(run/'candidate-evaluation.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_id');main(p.parse_args().run_id)
