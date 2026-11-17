"""Conservative PPE association with unique assignment and observable-time gating."""
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from app.geometry import overlap_fraction
from app.body_parts import associate_body_parts
from app.detail_inference import overlap_iou,valid_ppe_geometry


def direct_positive(person,item,kind,check_owner=True):
    """Positive PPE evidence may be visible even when face landmarks are not."""
    if item.confidence<.7:return False
    if check_owner and item.context_person_xyxy is not None and overlap_iou(person.xyxy,item.context_person_xyxy)<.8:return False
    return valid_ppe_geometry(person.xyxy,item.xyxy,kind)


def ambiguous_direct(people,item,kind):
    # Crop ownership is a processing link, not proof that an object belongs to
    # that person. Compare geometry against every plausible nearby worker.
    a,b,c,d=item.xyxy;costs=[]
    for person in people:
        if not direct_positive(person,item,kind,check_owner=False):continue
        x1,y1,x2,y2=person.xyxy
        costs.append(abs((a+c)/2-(x1+x2)/2)/(x2-x1)+abs((b+d)/2-y1)/(y2-y1))
    costs.sort()
    return len(costs)>1 and costs[1]-costs[0]<.05


def assign_direct_negative(people,items,kind):
    """Explicit trained no-PPE boxes, never absence inferred from a small crop."""
    costs=np.full((len(people),len(items)),1e6)
    for i,person in enumerate(people):
        x1,y1,x2,y2=person.xyxy
        for j,item in enumerate(items):
            if not direct_positive(person,item,kind) or ambiguous_direct(people,item,kind):continue
            a,b,c,d=item.xyxy
            costs[i,j]=abs((a+c)/2-(x1+x2)/2)/(x2-x1)+abs((b+d)/2-y1)/(y2-y1)+(1-item.confidence)*.2
    if not costs.size:return {}
    for j in range(len(items)):
        eligible=sorted(c for c in costs[:,j] if c<1e5)
        if len(eligible)>1 and eligible[1]-eligible[0]<.05:costs[:,j]=1e6
    rr,cc=linear_sum_assignment(costs)
    return {int(i):items[int(j)] for i,j in zip(rr,cc) if costs[i,j]<1e5}


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
            negative_class='bare_head' if kind=='helmet' else 'bare_torso'
            negatives=assign_direct_negative(people,[d for d in detections if d.class_name==negative_class and d.observed],kind)
            cost=np.full((len(people),len(items)),1e6)
            direct=np.zeros((len(people),len(items)),dtype=bool)
            for i,p in enumerate(people):
                x1,y1,x2,y2=p.xyxy; w=x2-x1; h=y2-y1
                region=(x1,y1-h*.05,x2,y1+h*.32) if kind=="helmet" else (x1,y1+h*.22,x2,y1+h*.72)
                observed_part=parts[p.key].get(part_name)
                if localized_mode:
                    if observed_part is not None:region=observed_part.xyxy
                for j,item in enumerate(items):
                    bx,by,ex,ey=item.xyxy; cx=(bx+ex)/2; cy=(by+ey)/2
                    if item.context_person_xyxy is not None and overlap_iou(p.xyxy,item.context_person_xyxy)<.8:continue
                    if localized_mode and (observed_part is None or item.context_person_xyxy is not None):
                        if direct_positive(p,item,kind) and not ambiguous_direct(people,item,kind):
                            direct[i,j]=True
                            cost[i,j]=abs(cx-(x1+x2)/2)/w+abs(cy-y1)/h+(1-item.confidence)*.2
                        continue
                    scale=(ex-bx)/w
                    if region[0]<=cx<=region[2] and region[1]<=cy<=region[3] and .08<=scale<=1.05 and overlap_fraction(item.xyxy,region)>.5:
                        cost[i,j]=abs(cx-(x1+x2)/2)/w+abs(cy-(region[1]+region[3])/2)/h + (1-item.confidence)*.2
            assigned={}
            if cost.size:
                # An actual item can support only one unambiguous person.
                for j in range(len(items)):
                    eligible=sorted(cost[i,j] for i in range(len(people)) if cost[i,j]<1e5)
                    if direct[:,j].any() and len(eligible)>1 and eligible[1]-eligible[0]<.05:cost[:,j]=1e6
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
                item=items[assigned[i]] if i in assigned else None
                positive_visible=bool(item is not None and direct_positive(p,item,kind) and crop.size and crop[::8,::8].mean()>25)
                negative=negatives.get(i)
                negative_visible=bool(negative is not None and crop.size and crop[::8,::8].mean()>25)
                conflicting=negative_visible and item is not None
                raw='unknown' if conflicting else "equipped" if item is not None and (observable or positive_visible) else ("missing" if observable or negative_visible else "unknown")
                key=(p.key,kind); now=frame.received_monotonic
                old=self.history.get(key)
                if old is None or now-old["last"]>self.gap or raw!=old["raw"]:
                    old={"raw":raw,"elapsed":0.,"hits":0,"last":now}
                else: old["elapsed"]+=max(0.,now-old["last"])
                old["last"]=now; old["hits"]+=1; self.history[key]=old
                threshold=self.clear if raw=="equipped" else self.confirm
                final=raw if raw!="unknown" and old["elapsed"]>=threshold and old["hits"]>=3 else "unknown"
                region=negative if negative_visible and not conflicting else observed_part if observed_part is not None else (item if positive_visible else None)
                basis=('conflicting_direct_ppe_states' if conflicting else 'direct_negative_ppe_unique_assignment' if negative_visible else 'direct_ppe_unique_assignment' if positive_visible and observed_part is None else
                    'head_or_torso_unique_assignment' if item is not None and observable else
                    'person_too_small_for_absence' if h<self.minimum else
                    'body_part_not_observed' if localized_mode and observed_part is None else
                    'sustained_visible_absence' if observable else 'person_not_observable')
                state[p.key][kind]={"state":final,"candidate":raw,"observed_s":round(old["elapsed"],3),"observable":(observable or positive_visible or negative_visible) and not conflicting,
                                    "absence_observable":observable,"basis":basis,
                                    "region_xyxy":list(region.xyxy) if region is not None else None,
                                    "region_confidence":region.confidence if region is not None else None,
                                    "region_basis":('detected_'+negative_class if negative_visible and not conflicting else (observed_part.localization_basis or "detected_"+part_name) if observed_part is not None else ('detected_'+kind if positive_visible else "unavailable"))}
        now=frame.received_monotonic
        self.history={k:v for k,v in self.history.items() if now-v["last"]<30}
        return state
