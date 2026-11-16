"""Incident-level matching; inputs require camera/type/subject and source-time intervals."""
import argparse
import json
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import numpy as np
from app.common import read_json,write_json


def score(truth,predictions,camera_hours,overlap_threshold=.3):
    if camera_hours<=0:raise ValueError("camera_hours_must_be_positive")
    for event in truth+predictions:
        for key in ("camera_id","type","subject_id","start_s","end_s"):
            if key not in event:raise ValueError("missing_event_field:"+key)
        if event["end_s"]<event["start_s"]:raise ValueError("negative_event_duration")
    costs=np.ones((len(truth),len(predictions)))
    for i,a in enumerate(truth):
        for j,b in enumerate(predictions):
            if any(a[k]!=b[k] for k in ("camera_id","type","subject_id")) or a.get("zone_id")!=b.get("zone_id"):continue
            intersection=max(0,min(a["end_s"],b["end_s"])-max(a["start_s"],b["start_s"]))
            union=max(a["end_s"],b["end_s"])-min(a["start_s"],b["start_s"])
            costs[i,j]=1-intersection/max(union,.001)
    matched=[]
    if costs.size:
        rows,cols=linear_sum_assignment(costs)
        matched=[(int(i),int(j)) for i,j in zip(rows,cols) if costs[i,j]<=1-overlap_threshold]
    tp=len(matched);fp=len(predictions)-tp;fn=len(truth)-tp
    return {"true_positive":tp,"false_positive":fp,"false_negative":fn,"truth_incidents":len(truth),"predicted_incidents":len(predictions),
        "precision":tp/(tp+fp) if tp+fp else None,"recall":tp/(tp+fn) if tp+fn else None,
        "false_alarms_per_camera_hour":fp/camera_hours,"camera_hours":camera_hours,"matches":matched,"interval_iou_threshold":overlap_threshold}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("truth");p.add_argument("predictions");p.add_argument("--camera-hours",type=float,required=True);p.add_argument("--output",required=True);a=p.parse_args()
    result=score(read_json(a.truth),read_json(a.predictions),a.camera_hours);write_json(a.output,result);print(json.dumps(result))
