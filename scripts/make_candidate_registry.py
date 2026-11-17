"""Create an offline registry; leave the operational registry unchanged."""
import argparse,copy
from app.common import ROOT,read_json,write_json,sha256


def candidate(run_id,kind):
    folder=ROOT/'outputs/training'/run_id;p=read_json(folder/'provenance.json');evaluation=read_json(folder/'candidate-evaluation.json')
    assert p['status']=='completed' and not p.get('user_video_used_for_training',True)
    assert sha256(ROOT/p['checkpoint'])==p['checkpoint_sha256']==evaluation['checkpoint_sha256']
    for split,digest in p['manifest_sha256'].items():assert sha256(ROOT/p['manifest_paths'][split])==digest
    names=['person'] if kind=='person' else ['helmet','safety_vest']
    return dict(run_id=run_id,path=p['checkpoint'],sha256=p['checkpoint_sha256'],classes=names,active_classes=names,enabled=True,
        evaluation_status='development_evaluated',initial_weights_sha256=p['initial_weights_sha256'],initial_weights_source=p['initial_weights_source'],
        manifest_sha256=p['manifest_sha256'],evaluation=evaluation,
        inference=dict(nms=evaluation['selected_nms'],confidence=.25 if kind=='person' else evaluation['selected_operating_point']['confidence'],
            head_basis=evaluation.get('head_selection_reason','native validation mAP50-95'),threshold_basis='existing person display threshold retained' if kind=='person' else evaluation['selected_operating_point']['basis']),
        license='AGPL-3.0; source dataset CC-BY-4.0 attribution also retained')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--person');parser.add_argument('--ppe');parser.add_argument('--helmet-only',action='store_true');parser.add_argument('--output',required=True);args=parser.parse_args()
    if args.helmet_only and not args.ppe:raise ValueError('helmet_model_required')
    operational=ROOT/'models/model_registry.json';original=sha256(operational);registry=copy.deepcopy(read_json(operational))
    output=(ROOT/args.output).resolve();assert output!=operational.resolve()
    for kind,run_id in [('person',args.person),('ppe',args.ppe)]:
        if run_id:
            names={'person'} if kind=='person' else {'helmet','safety_vest'}
            indices=[i for i,m in enumerate(registry['models']) if m.get('role','safety_detection')=='safety_detection' and names&set(m.get('active_classes',m['classes']))]
            assert len(indices)==1
            item=candidate(run_id,kind)
            if kind=='ppe' and args.helmet_only:
                registry['models'][indices[0]]['active_classes']=['safety_vest']
                item['active_classes']=['helmet']
                point=item['evaluation']['validation_heads'][item['evaluation']['selected_head']]['class_operating_points']['helmet']
                item['inference'].update(confidence=point['confidence'],threshold_basis='helmet F1 on unchanged native validation only')
                policy=read_json(ROOT/'models'/(run_id+'.operating-policy.json'),{})
                if policy and policy.get('status')=='accepted':
                    assert policy['checkpoint_sha256']==item['sha256']
                    item['inference'].update(confidence=policy['confidence'],threshold_basis=policy['basis'])
                registry['models'].insert(indices[0],item)
            else:registry['models'][indices[0]]=item
    registry.update(status='Offline candidate evaluation; existing label data only; distant CCTV acceptance unresolved',operational_parent_sha256=original)
    write_json(output,registry);assert sha256(operational)==original
