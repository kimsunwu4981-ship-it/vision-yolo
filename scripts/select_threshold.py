"""Validation-only detection threshold selection; no use of held-out test targets."""
import argparse
import numpy as np
from app.common import ROOT,read_json,write_json,sha256

if __name__=='__main__':
    from ultralytics import YOLO
    import torch
    p=argparse.ArgumentParser();p.add_argument('run_id');p.add_argument('--device',default='cpu');a=p.parse_args()
    torch.set_num_threads(4)
    run=ROOT/'outputs/training'/a.run_id;provenance=read_json(run/'provenance.json');model=YOLO(str(ROOT/provenance['checkpoint']))
    metrics=model.val(data=str(ROOT/'data/derived/ppe_v2/data.yaml'),split='val',imgsz=640,batch=8,device=a.device,
        workers=0,plots=False,verbose=False,project=str(run/'evaluation'),name='threshold-selection')
    curve=np.asarray(metrics.box.f1_curve);x=np.asarray(metrics.box.px);macro=curve.mean(axis=0);best=int(macro.argmax())
    report={'selection_split':'validation_only','validation_manifest_sha256':sha256(ROOT/'data/manifests/validation.json'),
        'criterion':'maximum macro bbox F1 at IoU 0.5 on validation confidence curve','confidence':float(x[best]),'macro_f1':float(macro[best]),
        'per_class_f1':{model.names[int(c)]:float(curve[i,best]) for i,c in enumerate(metrics.box.ap_class_index)},
        'device':a.device,'runtime_baseline_confidence':.25,'runtime_changed':False,
        'reason':'bbox F1 optimum is recorded for comparison; PPE/event confidence and time thresholds still require field incident validation; measured pipeline baseline retained'}
    write_json(ROOT/'models'/(a.run_id+'.threshold-selection.json'),report);print(report)
