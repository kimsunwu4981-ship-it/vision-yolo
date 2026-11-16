"""Adapt pose person confidence from real boxes; preserve official landmark weights.

No keypoint annotations exist in the authorized source data. This deliberately
uses detection loss ONLY and reports box metrics, never invented pose metrics.
"""
import argparse
import importlib.metadata
from app.common import ROOT, sha256, read_json, write_json, utc_now
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils.loss import E2ELoss
import torch


class BoxOnlyPoseTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = PoseModel(cfg, nc=1, data_kpt_shape=(17, 3), verbose=verbose)
        model.names = {0: 'person'}
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        super().set_model_attributes()
        self.model.criterion = E2ELoss(self.model)


def train(run_id, epochs):
    initial = ROOT/'models/yolo26n-pose.pt'
    assert sha256(initial) == 'eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9'
    output = ROOT/'outputs/training'/run_id
    if output.exists():
        raise ValueError('run_already_exists')
    manifests = {s: ROOT/'data/manifests/person-yolo26s-v4'/(s+'.json') for s in ('train','validation','test')}
    parent = dict(manifest_sha256={s:sha256(path) for s,path in manifests.items()},
        manifest_paths={s:str(path.relative_to(ROOT)) for s,path in manifests.items()})
    data = ROOT/'data/derived/person-yolo26s-v4/data.yaml'
    if not data.exists():
        raise FileNotFoundError('Reconstruct the person-yolo26s-v4 dataset from its committed manifests before training')
    last = 23
    initial_model = torch.load(initial, map_location='cpu', weights_only=False)['model']
    freeze = list(range(last)) + [f'{last}.{branch}' for branch, _ in
        initial_model.model[-1].named_children() if branch not in ('cv3', 'one2one_cv3')]
    del initial_model
    for branch in ('cv3', 'one2one_cv3'):
        for scale in range(3):
            for block in (0, 1):
                freeze.append(f'{last}.{branch}.{scale}.{block}')
    params = dict(model=str(initial), task='detect', data=str(data), epochs=epochs,
        imgsz=640, batch=16, device=0, workers=0, seed=53, deterministic=True,
        optimizer='AdamW', lr0=.0001, warmup_bias_lr=.0001, warmup_epochs=1,
        weight_decay=.0001, freeze=freeze, nms=False, patience=epochs,
        project=str(output.parent), name=run_id, exist_ok=False, cache=False,
        plots=True, amp=False, mosaic=0., mixup=0., scale=.1, translate=.05,
        close_mosaic=0, verbose=False)
    provenance = dict(run_id=run_id, status='running', started_at=utc_now(),
        initial_weights=str(initial.relative_to(ROOT)), initial_weights_sha256=sha256(initial),
        initial_weights_source='https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt',
        manifest_sha256=parent['manifest_sha256'], manifest_paths=parent['manifest_paths'],
        parameters=params, strategy='Real person-box supervision adapts classification outputs only. All landmark and box localization parameters and feature statistics remain frozen.',
        pose_supervision=False, keypoint_metrics=None, user_video_used_for_training=False,
        field_acceptance=False, gpu=torch.cuda.get_device_name(0),
        packages={name:importlib.metadata.version(name) for name in ('torch','torchvision','ultralytics')})
    trainer = BoxOnlyPoseTrainer(overrides=params)
    baseline = {}
    def capture(trainer):
        trainable = {n:p for n,p in trainer.model.named_parameters() if p.requires_grad}
        assert trainable and all(('.cv3.' in n or '.one2one_cv3.' in n) and '.2.' in n for n in trainable), list(trainable)
        baseline.update({n:p.detach().cpu().clone() for n,p in trainable.items()})
        provenance['trainable_parameters'] = {n:p.numel() for n,p in trainable.items()}
        write_json(output/'provenance.json', provenance)
    trainer.add_callback('on_pretrain_routine_end', capture)
    write_json(output/'provenance.json', provenance)
    try:
        trainer.train()
        best = output/'weights/best.pt'
        # Save an ordinary PoseModel with a pose inference task. Training-only
        # criterion is already stripped by Ultralytics; no custom model pickle.
        ckpt = torch.load(best, map_location='cpu', weights_only=False)
        model = ckpt.get('ema') or ckpt['model']
        changed = {n:float((p.detach().float().cpu()-baseline[n]).norm())
            for n,p in model.named_parameters() if n in baseline}
        assert any(v>0 for v in changed.values())
        official = torch.load(initial, map_location='cpu', weights_only=False)['model'].state_dict()
        current = model.state_dict()
        frozen_delta = max(float((current[n].float()-v.float()).abs().max())
            for n,v in official.items() if n not in baseline)
        assert frozen_delta == 0, 'frozen_pose_parameters_or_statistics_changed'
        ckpt['train_args']['task'] = 'pose'
        model.args['task'] = 'pose'
        torch.save(ckpt, best)
        provenance.update(status='completed', completed_at=utc_now(), checkpoint=str(best.relative_to(ROOT)),
            checkpoint_sha256=sha256(best), trained_parameter_l2_change=changed,
            frozen_parameters_and_buffers_max_absolute_change=frozen_delta,
            checkpoint_selection='highest validation person-box fitness; no pose metric used')
        write_json(output/'provenance.json', provenance)
        evaluation = dict(run_id=run_id, checkpoint_sha256=sha256(best),
            keypoint_metrics=None, limitation='No independent keypoint ground truth. Box metrics only.', splits={})
        for split in ('val','test'):
            validator = DetectionValidator(args=dict(model=str(best), task='detect', data=str(data),
                split=split, imgsz=640, batch=16, device=0, workers=0, nms=False,
                plots=False, project=str(output), name='box-evaluation-'+split))
            metrics = validator(model=str(best))
            evaluation['splits'][split] = {k:float(v) for k,v in metrics.items()}
        write_json(output/'candidate-evaluation.json', evaluation)
    except Exception as error:
        provenance.update(status='failed', error_type=type(error).__name__)
        write_json(output/'provenance.json', provenance)
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='body-pose-yolo26n-v2');parser.add_argument('--epochs',type=int,default=8)
    args=parser.parse_args();train(args.run_id,args.epochs)
