"""Unknown or absent observations suspend, never resolve, an active hazard."""
import uuid
from copy import deepcopy
from app.common import utc_now


class EventEngine:
    def __init__(self,max_gap_s=.75,cooldown_s=3.):
        self.states={}; self.max_gap=max_gap_s; self.cooldown=cooldown_s

    def update(self,frame,observations,model_hash,config_revision):
        changed=[]; now=frame.received_monotonic
        for obs in observations:
            state=self.states.get(obs.key)
            if state is None:
                if obs.condition is not True: continue
                state={"id":uuid.uuid4().hex,"key":obs.key,"type":obs.kind,"camera_id":frame.camera_id,
                    "session_id":frame.session_id,"tracks":obs.tracks,"zone_id":obs.zone_id,"rule_version":"1",
                    "model_hash":model_hash,"config_revision":config_revision,"status":"pending","acknowledged":False,
                    "started_at":frame.received_at_utc,"last_observed_at":frame.received_at_utc,"resolved_at":None,
                    "severity":obs.severity,"basis":obs.basis,"observation_state":"observed","image_id":None,"clip_id":None,
                    "_last":now,"_true":0.,"_false":0.,"_condition":True,"_resolved":None}
                self.states[obs.key]=state
            if state["status"]=="resolved":
                if obs.condition is True and now-state["_resolved"]>=self.cooldown:
                    del self.states[obs.key]
                    changed.extend(self.update(frame,[obs],model_hash,config_revision))
                continue
            previous=deepcopy(state)
            dt=now-state["_last"]
            if dt<0 or dt>self.max_gap or obs.condition is None:
                state["_true"]=0.; state["_false"]=0.; dt=0.
            if obs.condition is not None:
                state["last_observed_at"]=frame.received_at_utc
                state["basis"]=obs.basis
            state["observation_state"]="unknown" if obs.condition is None else "observed"
            if obs.condition is True:
                state["_true"]+=dt if state["_condition"] is True else 0
                state["_false"]=0.
                if state["status"]=="pending" and state["_true"]>=obs.confirm_s: state["status"]="active"
            elif obs.condition is False:
                state["_false"]+=dt if state["_condition"] is False else 0
                state["_true"]=0.
                if state["_false"]>=obs.clear_s:
                    state["status"]="resolved"; state["resolved_at"]=frame.received_at_utc; state["_resolved"]=now
            state["_last"]=now; state["_condition"]=obs.condition
            public={k:v for k,v in state.items() if not k.startswith("_")}
            # Persist bounded-rate updates and every transition.
            if previous["status"]!=state["status"] or now-state.get("_saved",-100)>1 or previous["observation_state"]!=state["observation_state"]:
                state["_saved"]=now; changed.append(public)
        observed_keys={o.key for o in observations}
        for key,state in list(self.states.items()):
            if state["camera_id"]==frame.camera_id and key not in observed_keys and state["status"]!="resolved":
                if state["observation_state"]!="unknown":
                    state["observation_state"]="unknown"; state["_true"]=state["_false"]=0.;state["_condition"]=None
                    changed.append({k:v for k,v in state.items() if not k.startswith("_")})
            if state["status"]=="resolved" and now-state["_resolved"]>max(30,self.cooldown): del self.states[key]
            elif now-state["_last"]>120:
                # DB retains unresolved hazards; evict inactive-session RAM only.
                del self.states[key]
        return changed

    def suspend_camera(self,camera_id):
        changes=[]
        for state in self.states.values():
            if state["camera_id"]==camera_id and state["status"]!="resolved" and state["observation_state"]!="unknown":
                state["observation_state"]="unknown";state["_true"]=state["_false"]=0.;state["_condition"]=None
                changes.append({k:v for k,v in state.items() if not k.startswith("_")})
        return changes
