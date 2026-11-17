"""Build an isolated, hash-checked registry for newly trained detail models."""
import argparse,copy
from app.common import ROOT,read_json,write_json,sha256


def main(output,helmet=False,head_state=None,vest_state=None):
    registry=copy.deepcopy(read_json(ROOT/'models/model_registry.json'))
    registry['models']=[m for m in registry['models'] if m.get('role') not in {'person_detail','ppe_detail','ppe_state'}]
    for record in registry['models']:
        if record.get('role')=='body_pose':
            record.setdefault('inference',{})['detail']=dict(min_person_height=48,median_max_person_height=180)
    runs=[('person-yolo26x-context-v8','person_detail','person')]
    if helmet:runs.append(('helmet-yolo26s-context-v6','ppe_detail','helmet'))
    for run_id,role,name in runs:
        folder=ROOT/'outputs/training'/run_id;p=read_json(folder/'provenance.json');e=read_json(folder/'candidate-evaluation.json')
        assert p['status']=='completed' and not p['user_video_used_for_training']
        assert sha256(ROOT/p['checkpoint'])==p['checkpoint_sha256']==e['checkpoint_sha256']
        for split,digest in p['manifest_sha256'].items():assert sha256(ROOT/p['manifest_paths'][split])==digest
        detail=dict(views=['median','median_flip'],view_imgsz={'median_flip':1600},microbatch=1,min_source_width=1000,min_person_height=24,max_person_height=180,tracking_low_confidence=.4) if role=='person_detail' else dict(min_person_height=32,max_person_height=180,crop_margin=.6,max_people=12)
        registry['models'].append(dict(run_id=run_id,path=p['checkpoint'],sha256=p['checkpoint_sha256'],role=role,classes=[name],active_classes=[name],enabled=True,
            evaluation_status='development_evaluated',initial_weights_sha256=p['initial_weights_sha256'],initial_weights_source=p['initial_weights_source'],manifest_sha256=p['manifest_sha256'],evaluation=e,
            inference=dict(confidence=.4 if role=='person_detail' else .7,nms=True if role=='person_detail' else e['selected_nms'],imgsz=1280 if role=='person_detail' else 640,detail=detail,
                threshold_basis='Small-object diagnostic operating policy; existing development data only; not field acceptance'),
            license='AGPL-3.0; source dataset CC-BY-4.0 attribution retained'))
    for run_id in (head_state,vest_state):
        if not run_id:continue
        folder=ROOT/'outputs/training'/run_id;p=read_json(folder/'provenance.json');e=read_json(folder/'candidate-evaluation.json')
        assert p['status']=='completed' and not p['user_video_used_for_training'] and sha256(ROOT/p['checkpoint'])==p['checkpoint_sha256']==e['checkpoint_sha256']
        for split,digest in p['manifest_sha256'].items():assert sha256(ROOT/p['manifest_paths'][split])==digest
        registry['models'].append(dict(run_id=run_id,path=p['checkpoint'],sha256=p['checkpoint_sha256'],role='ppe_state',classes=p['class_names'],active_classes=p['class_names'],enabled=True,
            evaluation_status='development_evaluated',initial_weights_sha256=p['initial_weights_sha256'],initial_weights_source=p['initial_weights_source'],manifest_sha256=p['manifest_sha256'],evaluation=e,
            inference=dict(confidence=.5,nms=e['selected_nms'],imgsz=640,detail=dict(min_person_height=32,max_person_height=180,crop_margins=[.6,1.,2.] if 'bare_head' in p['class_names'] else [.6],max_people=12,person_confidence=.25),
                threshold_basis='Detection .5; direct positive/explicit negative state .7 plus unique assignment and temporal confirmation. Bare-torso validation recall is zero at .85; .7 retains evidence with validation precision above .90.'),
            license='AGPL-3.0 model; existing source archive license retained in manifest source.json'))
    registry.update(status='Offline detail candidate; user videos excluded from training; field acceptance unresolved',operational_parent_sha256=sha256(ROOT/'models/model_registry.json'))
    target=(ROOT/output).resolve();assert target!=(ROOT/'models/model_registry.json').resolve();write_json(target,registry)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--helmet',action='store_true');parser.add_argument('--head-state');parser.add_argument('--vest-state');args=parser.parse_args();main(args.output,args.helmet,args.head_state,args.vest_state)
