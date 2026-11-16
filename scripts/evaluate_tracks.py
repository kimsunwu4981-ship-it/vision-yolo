"""Tracking evaluation requiring separately annotated per-frame ground truth."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from app.common import write_json


def load_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(truth,predictions):
    predicted={(r['camera_id'],r['frame_id']):r for r in predictions}
    # Input videos must have exactly one continuous session per camera.
    if len(predicted)!=len(predictions):raise ValueError('multiple_sessions_require_separate_runs')
    identity={};was_matched={};ever_matched=set();switches=fragments=matched=missed=false_positives=0
    gaps=defaultdict(list);last_observed={};ppe_total=ppe_unknown=0
    for row in sorted(truth,key=lambda r:(r['camera_id'],r['frame_id'])):
        camera=row['camera_id'];pred=predicted.get((camera,row['frame_id']),{});objects=row['objects']
        tracks=[t for t in pred.get('tracks',[]) if t.get('observed') and t.get('key')]
        cost=np.ones((len(objects),len(tracks)))
        for i,a in enumerate(objects):
            for j,b in enumerate(tracks):
                if a['class_name']!=b['class_name']:continue
                x=a['xyxy'];y=b['xyxy'];inter=max(0,min(x[2],y[2])-max(x[0],y[0]))*max(0,min(x[3],y[3])-max(x[1],y[1]))
                union=(x[2]-x[0])*(x[3]-x[1])+(y[2]-y[0])*(y[3]-y[1])-inter
                cost[i,j]=1-inter/max(1,union)
        pairs=[]
        if cost.size:
            rr,cc=linear_sum_assignment(cost);pairs=[(int(i),int(j)) for i,j in zip(rr,cc) if cost[i,j]<=.5]
        matched+=len(pairs);missed+=len(objects)-len(pairs);false_positives+=len(tracks)-len(pairs)
        hit={i for i,j in pairs}
        for i,j in pairs:
            key=(camera,objects[i]['id']);current=tracks[j]['key']
            if key in identity and identity[key]!=current:switches+=1
            if key in ever_matched and not was_matched.get(key,False):fragments+=1
            identity[key]=current;ever_matched.add(key)
            timestamp=float(row['source_pts'])
            if key in last_observed:gaps[camera].append(timestamp-last_observed[key])
            last_observed[key]=timestamp
        for i,obj in enumerate(objects):was_matched[(camera,obj['id'])]=i in hit
        for person in pred.get('ppe',{}).values():
            for item in person.values():ppe_total+=1;ppe_unknown+=item['state']=='unknown'
    return {"matched_observations":matched,"missed_observations":missed,"unmatched_predictions":false_positives,
        "id_switches":switches,"fragmentations":fragments,"ppe_unknown_ratio":ppe_unknown/ppe_total if ppe_total else None,
        "observation_gap_p95_s":{camera:float(np.percentile(values,95)) if values else None for camera,values in gaps.items()},
        "iou_threshold":.5,"ground_truth_frames":len(truth),"predicted_frames":len(predictions)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('truth');p.add_argument('tracks');p.add_argument('--output',required=True);a=p.parse_args()
    result=evaluate(load_lines(a.truth),load_lines(a.tracks));write_json(a.output,result);print(json.dumps(result))
