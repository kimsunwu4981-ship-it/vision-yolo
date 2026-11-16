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
