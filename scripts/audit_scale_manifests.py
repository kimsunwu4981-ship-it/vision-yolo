"""Audit source split isolation and exact files used by scale adaptation runs."""
import argparse,itertools
from app.common import ROOT,read_json,write_json,sha256,utc_now


def audit(run_id):
    folder=ROOT/'data/manifests'/run_id;summary={};sources={};groups={};verified={}
    def check(path,digest):
        path=ROOT/path
        if str(path) not in verified:verified[str(path)]=sha256(path)
        assert verified[str(path)]==digest, str(path)
    for split in ('train','validation','test'):
        path=folder/(split+'.json');rows=read_json(path)['records'];sources[split]=set();groups[split]=set();objects=0
        for row in rows:
            check(row['image'],row['image_sha256']);check(row['label'],row['derived_label_sha256'])
            for line in (ROOT/row['label']).read_text().splitlines():
                if not line.strip():continue
                cls,x,y,w,h=map(float,line.split())
                assert cls>=0 and cls==int(cls) and 0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1
                objects+=1
            assert not row.get('user_video_used',False)
            for source in row.get('source_records',[row]):
                check(source['image'],source['image_sha256']);check(source['label'],source['derived_label_sha256'])
                sources[split].add(source['image_sha256']);groups[split].add(source['group'])
        summary[split]=dict(rows=len(rows),unique_source_images=len(sources[split]),source_groups=len(groups[split]),
            annotation_rows_in_manifest_labels=objects,manifest_sha256=sha256(path))
    pairs={}
    for a,b in itertools.combinations(sources,2):
        pair=dict(image_hash_overlap=len(sources[a]&sources[b]),source_group_overlap=len(groups[a]&groups[b]))
        assert not any(pair.values()), (a,b,pair)
        pairs[a+'-'+b]=pair
    return dict(run_id=run_id,audited_at=utc_now(),splits=summary,pairwise_split_overlap=pairs,verified_files=len(verified),
        user_video_used_for_training=False,limitation='Verifies declared source isolation and file integrity; does not certify completeness or correctness of source annotations.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_ids',nargs='+');args=parser.parse_args()
    for run_id in args.run_ids:
        result=audit(run_id);write_json(ROOT/'data/manifests'/run_id/'audit.json',result);print(run_id,'verified',result['verified_files'])
