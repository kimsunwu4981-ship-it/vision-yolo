import json
from pathlib import Path
from app.common import ROOT,read_json,sha256,write_json

if __name__=='__main__':
    splits={name:read_json(ROOT/'data/manifests'/(name+'.json'))['records'] for name in ['train','validation','test']}
    for left,right in [('train','validation'),('train','test'),('validation','test')]:
        for key in ['group','image_sha256']:
            assert not {r[key] for r in splits[left]} & {r[key] for r in splits[right]},f'{key} leakage {left}/{right}'
    checked=0
    for records in splits.values():
        for row in records:
            assert sha256(ROOT/row['image'])==row['image_sha256']
            assert sha256(ROOT/row['label'])==row['derived_label_sha256']
            checked+=1
    result={'files_checked':checked,'group_and_exact_hash_split_disjoint':True,'source_site_time_independence':'not fully verifiable; see dataset audit'}
    write_json(ROOT/'outputs/diagnostics/manifest-validation.json',result);print(json.dumps(result))
