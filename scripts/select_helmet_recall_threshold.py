"""Select recall-weighted helmet F2 on native validation images only."""
import numpy as np
from app.common import ROOT,read_json,write_json,sha256


def main():
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(1)
    run=ROOT/'outputs/training/ppe-yolo26s-scale-v5';p=read_json(run/'provenance.json')
    checkpoint=ROOT/p['checkpoint'];assert sha256(checkpoint)==p['checkpoint_sha256']
    metrics=YOLO(str(checkpoint)).val(data=p['selection_data'],split='val',imgsz=640,batch=12,device=0,workers=0,quantize=16,nms=True,max_det=600,
        project=str(run/'evaluation'),name='helmet-f2-validation',plots=False,verbose=False)
    index=list(metrics.box.ap_class_index).index(0)
    precision=np.asarray(metrics.box.p_curve[index]);recall=np.asarray(metrics.box.r_curve[index])
    f2=5*precision*recall/np.maximum(4*precision+recall,1e-9);best=int(f2.argmax())
    report=dict(run_id=p['run_id'],checkpoint_sha256=sha256(checkpoint),confidence=float(metrics.box.px[best]),
        validation_precision=float(precision[best]),validation_recall=float(recall[best]),validation_F2=float(f2[best]),beta=2,
        basis='Native validation helmet F2 only; false negatives carry greater weight than F1. Development test was inspected before revising the objective, so it is not independent acceptance.',
        user_video_used_for_training=False)
    write_json(ROOT/'models/ppe-yolo26s-scale-v5.threshold-selection.json',report);print(report)


if __name__=='__main__':main()
