"""Conservative PPE association with unique assignment and observable-time gating."""
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from app.geometry import overlap_fraction
from app.body_parts import associate_body_parts


class PPE:
    def __init__(self,confirm_s=1.5,clear_s=.8,max_gap_s=.75,min_person_height=90):
        self.confirm=confirm_s; self.clear=clear_s; self.gap=max_gap_s; self.minimum=min_person_height
        self.history={}

    def update(self,frame,detections,available_classes):
        people=[d for d in detections if d.class_name=="person" and d.key and d.observed]
        state={p.key:{} for p in people}
        parts=associate_body_parts(people,detections)
        localized_mode={'head','torso'}.issubset(available_classes)
        for kind in ("helmet","safety_vest"):
            part_name='head' if kind=='helmet' else 'torso'
            items=[d for d in detections if d.class_name==kind and d.observed]
            cost=np.full((len(people),len(items)),1e6)
            for i,p in enumerate(people):
                x1,y1,x2,y2=p.xyxy; w=x2-x1; h=y2-y1
                region=(x1,y1-h*.05,x2,y1+h*.32) if kind=="helmet" else (x1,y1+h*.22,x2,y1+h*.72)
                observed_part=parts[p.key].get(part_name)
                if localized_mode:
                    if observed_part is None:continue
                    region=observed_part.xyxy
                for j,item in enumerate(items):
                    bx,by,ex,ey=item.xyxy; cx=(bx+ex)/2; cy=(by+ey)/2
                    scale=(ex-bx)/w
                    if region[0]<=cx<=region[2] and region[1]<=cy<=region[3] and .08<=scale<=1.05 and overlap_fraction(item.xyxy,region)>.5:
                        cost[i,j]=abs(cx-(x1+x2)/2)/w+abs(cy-(region[1]+region[3])/2)/h + (1-item.confidence)*.2
            assigned={}
            if cost.size:
                rr,cc=linear_sum_assignment(cost)
                assigned={int(i):int(j) for i,j in zip(rr,cc) if cost[i,j]<1e5}
            for i,p in enumerate(people):
                x1,y1,x2,y2=p.xyxy; h=y2-y1
                crop=frame.image[max(0,int(y1)):min(frame.source_height,int(y2)),max(0,int(x1)):min(frame.source_width,int(x2))]
                visible=(h>=self.minimum and x1>1 and x2<frame.source_width-1 and y1>1 and y2<frame.source_height-1 and crop.size>0 and crop[::8,::8].mean()>25)
                # Occlusion cannot be inferred reliably from bounding boxes; obvious overlap => unknown.
                if any(q is not p and overlap_fraction(p.xyxy,q.xyxy)>.35 for q in people): visible=False
                observable=visible and kind in available_classes
                observed_part=parts[p.key].get(part_name)
                if localized_mode and observed_part is None:observable=False
                raw="equipped" if i in assigned and observable else ("missing" if observable else "unknown")
                key=(p.key,kind); now=frame.received_monotonic
                old=self.history.get(key)
                if old is None or now-old["last"]>self.gap or raw!=old["raw"]:
                    old={"raw":raw,"elapsed":0.,"hits":0,"last":now}
                else: old["elapsed"]+=max(0.,now-old["last"])
                old["last"]=now; old["hits"]+=1; self.history[key]=old
                threshold=self.clear if raw=="equipped" else self.confirm
                final=raw if raw!="unknown" and old["elapsed"]>=threshold and old["hits"]>=3 else "unknown"
                state[p.key][kind]={"state":final,"candidate":raw,"observed_s":round(old["elapsed"],3),"observable":observable,
                                    "basis":"body_part_not_observed" if localized_mode and observed_part is None else ("head_or_torso_unique_assignment" if i in assigned else "sustained_visible_absence"),
                                    "region_xyxy":list(observed_part.xyxy) if observed_part is not None else None,
                                    "region_confidence":observed_part.confidence if observed_part is not None else None,
                                    "region_basis":(observed_part.localization_basis or "detected_"+part_name) if observed_part is not None else "unavailable"}
        now=frame.received_monotonic
        self.history={k:v for k,v in self.history.items() if now-v["last"]<30}
        return state
