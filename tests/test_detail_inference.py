from types import SimpleNamespace
import numpy as np
import torch
from app.inference import Detector
from app.detail_inference import detail_people,context_ppe,merge_class,valid_ppe_geometry
from app.types import Detection
from app.ppe import PPE


class PersonModel:
    def predict(self,images,**kwargs):
        assert len(images)<=4
        return [SimpleNamespace(orig_shape=im.shape[:2],names={0:'person'},boxes=SimpleNamespace(data=torch.tensor([[20,10,50,80,.8,0]]))) for im in images]


def test_detail_flip_restored_once_and_same_view_boxes_merged(frame_factory):
    frames=[frame_factory(.2,width=1280,height=720,frame_id=8),frame_factory(.3,width=720,height=1280,frame_id=9)]
    output=[[],[]];detector=Detector({'microbatch':4});detector.device='cpu'
    detail_people(detector,PersonModel(),{'confidence':.25,'imgsz':1280,'nms':True,'detail':{}},frames,output)
    assert sorted(d.xyxy for d in output[0])==[(20,10,50,80),(1230,10,1260,80)]
    assert not output[1]
    assert all(d.last_observed_frame==8 and d.last_observed_monotonic==.2 for d in output[0])


def test_ppe_crop_coordinates_and_owner_follow_each_frame(frame_factory):
    class Model:
        def predict(self,images,**kwargs):
            assert kwargs['rect'] is False and len(images)<=4
            return [SimpleNamespace(orig_shape=im.shape[:2],names={0:'helmet'},boxes=SimpleNamespace(data=torch.tensor([[40,35,55,47,.9,0]]))) for im in images]
    frames=[frame_factory(.2,frame_id=8),frame_factory(.3,frame_id=9)]
    people=[Detection('person',(100,100,150,180),.9),Detection('person',(400,200,450,280),.9)]
    output=[[people[0]],[people[1]]];detector=Detector({'microbatch':4});detector.device='cpu'
    context_ppe(detector,Model(),{'confidence':.5,'imgsz':640,'nms':True,'active_classes':['helmet'],'detail':{}},frames,output)
    assert output[0][1].xyxy==(108,103,123,115)
    assert output[1][1].xyxy==(408,203,423,215)
    assert output[0][1].context_person_xyxy==people[0].xyxy
    assert output[1][1].last_observed_frame==9


def test_merge_keeps_different_objects_and_classes():
    a=Detection('person',(10,10,60,90),.9);duplicate=Detection('person',(11,11,61,91),.8)
    b=Detection('person',(80,10,130,90),.8);helmet=Detection('helmet',(10,10,60,90),.9)
    assert merge_class([a,duplicate,b,helmet],'person')==[helmet,a,b]


def test_view_scales_do_not_mix_in_one_batch(frame_factory):
    calls=[]
    class Model(PersonModel):
        def predict(self,images,**kwargs):
            calls.append((kwargs['imgsz'],len(images)))
            return super().predict(images,**kwargs)
    frames=[frame_factory(.2,width=1280,height=720)];detector=Detector({'microbatch':4});detector.device='cpu'
    detail_people(detector,Model(),{'confidence':.25,'imgsz':1280,'nms':True,'detail':{'view_imgsz':{'flip':1600}}},frames,[[]])
    assert calls==[(1280,1),(1280,1),(1600,1)]


def test_median_view_keeps_original_coordinates_and_restores_flip(frame_factory):
    frame=frame_factory(.2,width=1280,height=720);detector=Detector({'microbatch':4});detector.device='cpu';out=[[]]
    detail_people(detector,PersonModel(),{'confidence':.25,'imgsz':1280,'nms':True,'detail':{'views':['median','median_flip'],'view_imgsz':{'median_flip':1600}}},[frame],out)
    assert sorted(d.xyxy for d in out[0])==[(20,10,50,80),(1230,10,1260,80)]


def test_detail_floor_does_not_emit_weak_structure_boxes(frame_factory):
    class Model(PersonModel):
        def predict(self,images,**kwargs):
            assert kwargs['conf']==.4
            results=super().predict(images,**kwargs)
            for result in results:result.boxes.data[0,4]=.3
            return results
    detector=Detector({'microbatch':4,'tracking_low_confidence':.1});detector.device='cpu';out=[[]]
    detail_people(detector,Model(),{'confidence':.4,'imgsz':1280,'nms':True,'detail':{'views':['median'],'tracking_low_confidence':.4}},[frame_factory(.2,width=1280,height=720)],out)
    assert not out[0]


def test_small_people_can_have_direct_positive_ppe_without_inventing_absence(frame_factory):
    people=[Detection('person',(x,100,x+40,170),.9,key=str(x)) for x in (50,200,350)]
    helmets=[Detection('helmet',(p.xyxy[0]+10,100,p.xyxy[0]+25,112),.9,context_person_xyxy=p.xyxy) for p in people]
    ppe=PPE();classes={'person','helmet','safety_vest','head','torso'}
    for t in (0,.4,.9):state=ppe.update(frame_factory(t),people+helmets,classes)
    for p,h in zip(people,helmets):
        assert state[p.key]['helmet']['state']=='equipped'
        assert state[p.key]['helmet']['region_xyxy']==list(h.xyxy)
        assert not state[p.key]['helmet']['absence_observable']
        assert state[p.key]['safety_vest']['state']=='unknown'
    # Direct PPE evidence must not become an invented head region next frame.
    state=ppe.update(frame_factory(1.),people,classes)
    assert all(s['helmet']['state']=='unknown' and s['helmet']['region_xyxy'] is None for s in state.values())


def test_direct_ppe_rejects_ambiguous_person_and_weak_or_wrong_owner(frame_factory):
    people=[Detection('person',(50,100,90,170),.9,key=k) for k in ('A','B')]
    helmet=Detection('helmet',(60,100,75,112),.9)
    classes={'person','helmet','head','torso'}
    for objects in (people+[helmet],[people[0],Detection('helmet',helmet.xyxy,.69)],
                    [people[0],Detection('helmet',helmet.xyxy,.9,context_person_xyxy=(300,100,340,170))]):
        ppe=PPE(clear_s=.1)
        for t in (0,.1,.2):state=ppe.update(frame_factory(t),objects,classes)
        assert all(s['helmet']['state']=='unknown' for s in state.values())


def test_explicit_negative_boxes_localize_small_people_without_pose(frame_factory):
    person=Detection('person',(100,100,145,180),.9,key='P')
    head=Detection('bare_head',(109,103,124,117),.92,context_person_xyxy=person.xyxy)
    torso=Detection('bare_torso',(105,120,139,158),.93,context_person_xyxy=person.xyxy)
    ppe=PPE();classes={'person','helmet','safety_vest','head','torso','bare_head','bare_torso'}
    for t in (0,.5,1.,1.6):state=ppe.update(frame_factory(t),[person,head,torso],classes)
    assert state['P']['helmet']['state']==state['P']['safety_vest']['state']=='missing'
    assert state['P']['helmet']['region_xyxy']==list(head.xyxy)
    assert state['P']['safety_vest']['region_xyxy']==list(torso.xyxy)
    assert not state['P']['helmet']['absence_observable']
    assert state['P']['helmet']['basis']=='direct_negative_ppe_unique_assignment'
    state=ppe.update(frame_factory(1.7),[person],classes)
    assert state['P']['helmet']['state']=='unknown' and state['P']['helmet']['region_xyxy'] is None


def test_conflicting_positive_and_negative_ppe_is_unknown(frame_factory):
    person=Detection('person',(100,100,145,180),.9,key='P');box=(109,103,124,117)
    items=[person,Detection('helmet',box,.93,context_person_xyxy=person.xyxy),Detection('bare_head',box,.94,context_person_xyxy=person.xyxy)]
    ppe=PPE();classes={'person','helmet','head','torso','bare_head'}
    for t in (0,.5,1.,1.6):state=ppe.update(frame_factory(t),items,classes)
    assert state['P']['helmet']['state']=='unknown'
    assert state['P']['helmet']['basis']=='conflicting_direct_ppe_states'
    assert not state['P']['helmet']['observable']


def test_explicit_negative_rejects_weak_or_ambiguous_boxes(frame_factory):
    a=Detection('person',(100,100,145,180),.9,key='A');b=Detection('person',a.xyxy,.9,key='B')
    for items in ([a,Detection('bare_head',(109,103,124,117),.69)],[a,b,Detection('bare_head',(109,103,124,117),.99)]):
        ppe=PPE()
        for t in (0,.5,1.,1.6):state=ppe.update(frame_factory(t),items,{'person','helmet','head','torso','bare_head'})
        assert all(s['helmet']['state']=='unknown' for s in state.values())


def test_crop_owner_does_not_hide_an_ambiguous_neighbor(frame_factory):
    a=Detection('person',(100,100,140,180),.9,key='A');b=Detection('person',(120,100,160,180),.9,key='B')
    for kind in ('helmet','bare_head'):
        item=Detection(kind,(124,102,136,115),.99,context_person_xyxy=a.xyxy);ppe=PPE()
        for t in (0,.5,1.,1.6):state=ppe.update(frame_factory(t),[a,b,item],{'person','helmet','head','torso','bare_head'})
        assert all(s['helmet']['state']=='unknown' for s in state.values())


def test_crop_geometry_rejects_foot_level_helmet_without_moving_boxes():
    person=(100,100,145,180)
    assert valid_ppe_geometry(person,(110,102,125,116),'helmet')
    assert not valid_ppe_geometry(person,(110,160,125,175),'helmet')
    assert valid_ppe_geometry(person,(105,120,140,158),'bare_torso')
    assert not valid_ppe_geometry(person,(105,90,140,112),'bare_torso')


def test_multiple_contexts_keep_one_actual_box_per_owner(frame_factory):
    class Model:
        def predict(self,images,**kwargs):
            assert len(images)<=4
            result=[]
            for im in images:
                # person=(100,100,150,180); .4 margin starts at 68, .5 at 60.
                row=[34,34,45,48,.8,0] if im.shape[0]==144 else [70,42,82,56,.9,0]
                result.append(SimpleNamespace(orig_shape=im.shape[:2],names={0:'helmet'},boxes=SimpleNamespace(data=torch.tensor([row]))))
            return result
    detector=Detector({'microbatch':4});detector.device='cpu';out=[[Detection('person',(100,100,150,180),.9)]]
    context_ppe(detector,Model(),{'confidence':.5,'imgsz':640,'nms':True,'active_classes':['helmet'],'detail':{'crop_margins':[.4,.5]}},[frame_factory(.2)],out)
    helmets=[d for d in out[0] if d.class_name=='helmet']
    assert len(helmets)==1 and helmets[0].xyxy==(130,102,142,116)
