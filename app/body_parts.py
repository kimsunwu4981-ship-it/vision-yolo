"""Associate current-image anatomical regions, never fixed person-box fractions."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from app.geometry import overlap_fraction


def associate_body_parts(people,detections):
    result={p.key:{} for p in people}
    for kind in ('head','torso'):
        parts=[d for d in detections if d.class_name==kind and d.observed]
        cost=np.full((len(people),len(parts)),1e6)
        for i,person in enumerate(people):
            x1,y1,x2,y2=person.xyxy;w=x2-x1;h=y2-y1
            for j,part in enumerate(parts):
                a,b,c,d=part.xyxy;cx=(a+c)/2;cy=(b+d)/2
                fraction=overlap_fraction(part.xyxy,person.xyxy);area=(c-a)*(d-b)/(w*h)
                if not (x1<=cx<=x2 and y1<=cy<=y2 and fraction>=.8 and .001<=area<(.5 if kind=='head' else .95)):continue
                cost[i,j]=1-fraction+.2*(1-part.confidence)+.05*abs(cx-(x1+x2)/2)/w
        if not cost.size:continue
        rr,cc=linear_sum_assignment(cost)
        for i,j in zip(rr,cc):
            if cost[i,j]>=1e5:continue
            alternatives=[cost[k,j] for k in range(len(people)) if k!=i and cost[k,j]<1e5]
            if alternatives and min(alternatives)-cost[i,j]<.03:continue
            result[people[i].key][kind]=parts[j]
    return result
