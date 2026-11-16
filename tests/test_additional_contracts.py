import time
from app.types import Detection,RuleObservation
from app.tracking import TrackerBank
from app.events import EventEngine
from scripts.evaluate_events import score


def test_track_no_reuse_after_ttl_with_continuous_empty_frames(frame_factory):
    tracker=TrackerBank(ttl_s=.4)
    first=tracker.update(frame_factory(0),[Detection("person",(100,40,200,400),.9)])[0]
    for t in [.1,.2,.3,.5,.7]:assert tracker.update(frame_factory(t),[])==[]
    tracker.update(frame_factory(.8),[Detection("person",(100,40,200,400),.9)])
    second=tracker.update(frame_factory(.9),[Detection("person",(100,40,200,400),.9)])[0]
    assert second.track_id!=first.track_id


def test_event_subject_matching_not_just_camera():
    truth=[dict(camera_id="c",type="forklift_proximity",subject_id="person-A",start_s=0,end_s=5)]
    wrong=[dict(camera_id="c",type="forklift_proximity",subject_id="person-B",start_s=0,end_s=5)]
    result=score(truth,wrong,2)
    assert result["recall"]==0 and result["false_alarms_per_camera_hour"]==.5
    result=score(truth,truth,2)
    assert result["precision"]==1 and result["recall"]==1


def test_unknown_does_not_refresh_last_observation(frame_factory):
    engine=EventEngine()
    first=engine.update(frame_factory(0),[RuleObservation("p","ppe_violation",True,confirm_s=0)],"m","c")[0]
    engine.update(frame_factory(5),[RuleObservation("p","ppe_violation",None)],"m","c")
    assert engine.states["p"]["last_observed_at"]==first["last_observed_at"]
    assert engine.states["p"]["status"]=="active"


def test_low_score_observation_keeps_existing_track_but_cannot_spawn(frame_factory):
    tracker=TrackerBank(ttl_s=1.,high_confidence=.25,low_confidence=.1)
    first=tracker.update(frame_factory(0),[Detection('person',(100,40,200,400),.9)])[0]
    low=tracker.update(frame_factory(.1),[Detection('person',(101,40,201,400),.15)])
    assert len(low)==1 and low[0].key==first.key and low[0].observed
    assert low[0].last_observed_monotonic==.1
    separate=TrackerBank(ttl_s=1.,high_confidence=.25,low_confidence=.1)
    assert separate.update(frame_factory(0),[Detection('person',(100,40,200,400),.15)])==[]


def test_detector_class_routing_and_separate_ppe_threshold(frame_factory):
    from types import SimpleNamespace
    import torch
    from app.inference import Detector
    frame=frame_factory(0)
    class Model:
        def __init__(self,rows,names):self.rows=rows;self.names=names;self.calls=[]
        def predict(self,images,**kwargs):
            self.calls.append(kwargs)
            return [SimpleNamespace(orig_shape=(frame.source_height,frame.source_width),
                boxes=SimpleNamespace(data=torch.tensor(self.rows)),names=self.names) for _ in images]
    person=Model([[100,40,200,400,.15,0]],{0:'person'})
    ppe=Model([[100,40,200,400,.99,0],[120,40,150,70,.3,1],[120,80,170,200,.8,2]],
        {0:'person',1:'helmet',2:'safety_vest'})
    detector=Detector({'confidence':.25,'tracking_low_confidence':.1});detector.device='cpu'
    detector.models=[person,ppe]
    detector.policies=[{'active_classes':['person'],'confidence':.25,'nms':False},
        {'active_classes':['helmet','safety_vest'],'confidence':.5365,'nms':True}]
    result=detector.predict(frame)
    assert [d.class_name for d in result]==['person','safety_vest']
    assert person.calls[0]['conf']==.1 and person.calls[0]['nms'] is False
    assert ppe.calls[0]['conf']==.5365 and ppe.calls[0]['nms'] is True
