"""Summarize observed boxes and states; never label them as ground-truth recall."""
import argparse,json,csv
from collections import Counter
import numpy as np
from app.common import ROOT,write_json


def summarize(path):
    path=ROOT/path;rows=[json.loads(line) for line in (path/'tracks.jsonl').read_text(encoding='utf-8').splitlines()];objects=Counter();states={k:Counter() for k in ('helmet','safety_vest')};candidates={k:Counter() for k in states};people=[];three=0
    three_regions=0
    for row in rows:
        counts=Counter(d['class_name'] for d in row['tracks'] if d['observed']);objects.update(counts);people.append(counts['person'])
        three+=int(any(p['helmet']['state']=='missing' and p['safety_vest']['state']=='missing' and p['helmet']['region_xyxy'] and p['safety_vest']['region_xyxy'] for p in row['ppe'].values()))
        three_regions+=int(any(all(p[k]['region_xyxy'] and (p[k]['state']=='missing' or p[k]['candidate']=='equipped') for k in states) for p in row['ppe'].values()))
        for person in row['ppe'].values():
            for kind in states:states[kind][person[kind]['state']]+=1;candidates[kind][person[kind]['candidate']]+=1
    performance=list(csv.DictReader((path/'performance.csv').open()));times=[float(r['processing_ms']) for r in performance]
    result=dict(frames=len(rows),person_frames=sum(n>0 for n in people),multi_person_frames=sum(n>=2 for n in people),maximum_simultaneous_person_boxes=max(people,default=0),person_observations=sum(people),
        observations=dict(objects),ppe_states={k:dict(v) for k,v in states.items()},ppe_candidates={k:dict(v) for k,v in candidates.items()},three_box_frames=three,person_with_head_and_torso_boxes_frames=three_regions,
        processing_ms=dict(mean=float(np.mean(times)),p95=float(np.percentile(times,95))),field_accuracy_evaluated=False,
        limitation='Counts include prediction errors and are not accuracy/recall or unique-person counts. Processing excludes decode/load/render; concurrent workloads, if any, affect timing.')
    write_json(path/'test-report.json',result);print(result,flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path');summarize(p.parse_args().path)
