import math
from app.geometry import foot,polygon_distance,overlap_fraction,projected_distance
from app.types import RuleObservation


class Rules:
    def __init__(self,config):
        self.config=config
        self.zone_state={}
        self.proximity_state={}

    def update(self,frame,tracks,ppe,zones,camera):
        observations=[]; now=frame.received_monotonic
        people=[d for d in tracks if d.class_name=="person" and d.key and d.observed]
        forklifts=[d for d in tracks if d.class_name=="forklift" and d.key and d.observed]
        for person in people:
            px,py=foot(person.xyxy); point=(px/frame.source_width,py/frame.source_height)
            required=set(camera.get("required_ppe",["helmet","safety_vest"]))
            for zone in zones:
                key=f"{person.key}:zone:{zone['id']}"
                location=point
                if zone.get("anchor","foot")=="center":
                    x1,y1,x2,y2=person.xyxy; location=((x1+x2)/2/frame.source_width,(y1+y2)/2/frame.source_height)
                distance=polygon_distance(location,zone["points"])
                old=self.zone_state.get(key,{"inside":False,"dwell":0.,"last":now})
                gap=now-old["last"]>self.config.get("max_observation_gap_s",.75)
                if gap: old={"inside":False,"dwell":0.,"last":now}
                margin=zone.get("margin",.005)
                inside=distance>=(-margin if old["inside"] else margin)
                dt=max(0,now-old["last"])
                dwell=old["dwell"]+dt if inside and old["inside"] else 0.
                transition="entry" if inside and not old["inside"] else ("exit" if not inside and old["inside"] else "stay")
                self.zone_state[key]={"inside":inside,"dwell":dwell,"last":now}
                if inside: required.update(zone.get("required_ppe",[]))
                observations.append(RuleObservation(key,"zone_intrusion",inside,[person.key],zone["id"],
                    {"transition":transition,"dwell_s":round(dwell,3),"anchor":zone.get("anchor","foot"),"signed_normalized_distance":distance},
                    "critical",zone.get("enter_s",1.),zone.get("exit_s",.8)))
            for kind in required:
                status=ppe.get(person.key,{}).get(kind,{}).get("state","unknown")
                observations.append(RuleObservation(f"{person.key}:ppe:{kind}","ppe_violation",
                    None if status=="unknown" else status=="missing",[person.key],basis={"item":kind,"state":status},
                    confirm_s=camera.get("ppe_event_confirm_s",.5),clear_s=camera.get("ppe_event_clear_s",1.)))
            for forklift in forklifts:
                key=f"{person.key}:near:{forklift.key}"; fx,fy=foot(forklift.xyxy)
                relative=math.hypot(px-fx,py-fy)/max(1,person.xyxy[3]-person.xyxy[1])
                overlap=overlap_fraction(person.xyxy,forklift.xyxy)
                calibration=camera.get("calibration")
                basis={"relative_person_height_distance":relative,"person_overlap":overlap,"driver_exclusion":False}
                old=self.proximity_state.get(key)
                if old and 0<now-old["time"]<self.config.get("max_observation_gap_s",.75): basis["approaching"]=relative<old["distance"]-.03
                self.proximity_state[key]={"distance":relative,"time":now}
                near=relative<self.config.get("proximity_person_heights",1.3) or overlap>.05
                if calibration and calibration.get("validated") and calibration.get("unit")=="metres":
                    metres=projected_distance((px,py),(fx,fy),calibration["homography"])
                    basis["ground_distance_m"]=metres; near=metres<calibration.get("threshold_m",2.)
                observations.append(RuleObservation(key,"forklift_proximity",near,[person.key,forklift.key],basis=basis,
                    severity="critical",confirm_s=self.config.get("proximity_confirm_s",.5),clear_s=1.))
        self.zone_state={k:v for k,v in self.zone_state.items() if now-v["last"]<30}
        self.proximity_state={k:v for k,v in self.proximity_state.items() if now-v["time"]<30}
        return observations
