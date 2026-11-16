import time
import numpy as np
import pytest
from app.geometry import transform_boxes,letterbox_matrix,validate_box,validate_polygon,projected_distance
from app.capture import LatestSlot
from app.events import EventEngine
from app.types import RuleObservation,Detection
from app.ppe import PPE
from app.rules import Rules
from app.tracking import TrackerBank


@pytest.mark.parametrize("matrix",[letterbox_matrix(1920,1080,640,640),np.array([[1,0,-50],[0,1,-100],[0,0,1]]),np.array([[0,-1,1080],[1,0,0],[0,0,1]])])
def test_coordinate_round_trip(matrix):
    original=np.array([[100.,200.,600.,800.]])
    np.testing.assert_allclose(transform_boxes(transform_boxes(original,matrix),np.linalg.inv(matrix)),original,atol=1e-6)

@pytest.mark.parametrize("box",[(0,0,0,2),(0,0,-1,2),(0,0,641,2),(0,0,float('nan'),2),(-1,0,20,20)])
def test_reject_invalid_boxes(box):
    with pytest.raises(ValueError):validate_box(box,640,480)

def test_polygon_self_intersection_and_calibration():
    with pytest.raises(ValueError):validate_polygon([[0,0],[1,1],[0,1],[1,0]])
    assert projected_distance((0,0),(100,0),[[.01,0,0],[0,.01,0],[0,0,1]])==pytest.approx(1)

def test_latest_frame_bounded(frame_factory):
    slot=LatestSlot()
    for i in range(100):slot.put(frame_factory(frame_id=i))
    assert slot.size()==1 and slot.dropped==99 and slot.take().frame_id==99 and slot.take() is None

def test_event_unknown_gap_ack_independence(frame_factory):
    engine=EventEngine(max_gap_s=.75)
    def update(t,condition):return engine.update(frame_factory(t),[RuleObservation("p1","ppe_violation",condition,confirm_s=.5,clear_s=.4)],"model","config")
    assert update(0,True)[0]["status"]=="pending"
    update(.3,True);assert update(.6,True)[0]["status"]=="active"
    update(10,None);assert engine.states["p1"]["status"]=="active"
    update(10.2,False);assert engine.states["p1"]["status"]=="active"
    update(10.7,False);assert engine.states["p1"]["status"]=="resolved"
    assert engine.states["p1"]["acknowledged"] is False

def test_no_accumulation_across_disconnect_and_per_person(frame_factory):
    engine=EventEngine(max_gap_s=.5)
    for t in [0,.2,10,10.2]:
        engine.update(frame_factory(t),[RuleObservation("p1","zone_intrusion",True,confirm_s=.5)],"m","c")
    assert engine.states["p1"]["status"]=="pending"
    engine.update(frame_factory(10.3),[RuleObservation("p2","zone_intrusion",True,confirm_s=0)],"m","c")
    assert engine.states["p2"]["status"]=="active"
    assert engine.states["p1"]["observation_state"]=="unknown"

def test_unique_ppe_and_unknown_on_dark_small_occluded(frame_factory,person):
    ppe=PPE(confirm_s=.2,clear_s=.2)
    helmet=Detection("helmet",(125,45,175,105),.9)
    for t in [0,.1,.3]:result=ppe.update(frame_factory(t),[person,helmet],{"person","helmet","safety_vest"})
    assert result[person.key]["helmet"]["state"]=="equipped"
    assert result[person.key]["safety_vest"]["state"]=="missing"
    result=ppe.update(frame_factory(.4,brightness=5),[person,helmet],{"helmet","safety_vest"})
    assert result[person.key]["helmet"]["state"]=="unknown"
    second=Detection("person",(140,50,240,400),.9,2,key="cam1:session1:person:2")
    result=ppe.update(frame_factory(.5),[person,second,helmet],{"helmet","safety_vest"})
    assert all(p["helmet"]["state"]=="unknown" for p in result.values())

def test_carried_helmet_not_equipped(frame_factory,person):
    ppe=PPE(confirm_s=.2,clear_s=.2)
    hand_helmet=Detection("helmet",(140,270,185,310),.95)
    for t in [0,.1,.3]:result=ppe.update(frame_factory(t),[person,hand_helmet],{"helmet"})
    assert result[person.key]["helmet"]["state"]=="missing"
    assert result[person.key]["safety_vest"]["state"]=="unknown"

def test_tracker_camera_session_class_isolation_and_expiry(frame_factory):
    tracker=TrackerBank(ttl_s=.6)
    keys=[]
    for cid in ["cam1","cam2","cam3","cam4"]:
        tracks=tracker.update(frame_factory(0,camera=cid),[Detection("person",(100,50,200,400),.9),Detection("forklift",(100,50,200,400),.9)])
        assert {t.class_name for t in tracks}=={"person","forklift"}
        keys.extend(t.key for t in tracks)
    assert len(set(keys))==8
    assert tracker.update(frame_factory(.1),[])==[]
    tracker.update(frame_factory(3,session="reconnected"),[Detection("person",(100,50,200,400),.9)])
    assert all(key[1]=="reconnected" for key in tracker.banks if key[0]=="cam1")

def test_zone_dwell_gap_and_forklift_driver_not_excluded(frame_factory,person):
    rules=Rules({"max_observation_gap_s":.75})
    zone={"id":"z1","points":[[.1,.1],[.9,.1],[.9,.9],[.1,.9]],"enter_s":.5}
    forklift=Detection("forklift",(90,40,220,420),.9,1,key="cam1:session1:forklift:1")
    for t in [0,.2,.4]:obs=rules.update(frame_factory(t),[person,forklift],{},[zone],{})
    z=next(o for o in obs if o.kind=="zone_intrusion");assert z.basis["dwell_s"]==pytest.approx(.4)
    prox=next(o for o in obs if o.kind=="forklift_proximity");assert prox.condition and not prox.basis["driver_exclusion"]
    assert "ground_distance_m" not in prox.basis
    obs=rules.update(frame_factory(5),[person],{},[zone],{})
    assert next(o for o in obs if o.kind=="zone_intrusion").basis["dwell_s"]==0

def test_zone_exit_and_reentry(frame_factory):
    rules=Rules({});zone={"id":"z","points":[[.2,.2],[.8,.2],[.8,.8],[.2,.8]]}
    def check(t,b):
        p=Detection("person",b,.9,1,key="p");return rules.update(frame_factory(t),[p],{},[zone],{"required_ppe":[]})[0]
    assert check(0,(200,100,300,300)).basis["transition"]=="entry"
    assert check(.2,(500,100,600,450)).basis["transition"]=="exit"
    assert check(.4,(200,100,300,300)).basis["transition"]=="entry"
