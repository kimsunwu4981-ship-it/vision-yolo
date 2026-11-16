from app.types import Detection
from app.body_parts import associate_body_parts
from app.ppe import PPE


def test_detected_head_can_move_below_fixed_top_band(frame_factory):
    person=Detection('person',(100,100,500,400),.9,1,key='worker')
    head=Detection('head',(105,200,155,250),.9)
    torso=Detection('torso',(200,150,380,310),.9)
    helmet=Detection('helmet',(108,201,150,225),.9)
    ppe=PPE(confirm_s=.1,clear_s=.1)
    for t in (0,.1,.2):state=ppe.update(frame_factory(t),[person,head,torso,helmet],{'person','helmet','safety_vest','head','torso'})
    assert state['worker']['helmet']['state']=='equipped'
    assert state['worker']['helmet']['region_xyxy']==list(head.xyxy)
    assert state['worker']['safety_vest']['state']=='missing'
    assert state['worker']['safety_vest']['region_xyxy']==list(torso.xyxy)
    # A fresh person observation alone must not reuse the prior head/torso box.
    unknown=ppe.update(frame_factory(.3),[person],{'person','helmet','safety_vest','head','torso'})
    assert all(s['state']=='unknown' and s['region_xyxy'] is None for s in unknown['worker'].values())


def test_body_parts_are_unique_per_person():
    a=Detection('person',(10,10,110,410),.9,key='A');b=Detection('person',(210,10,310,410),.9,key='B')
    ah=Detection('head',(30,20,70,70),.9);bh=Detection('head',(230,20,270,70),.9)
    result=associate_body_parts([a,b],[ah,bh])
    assert result['A']['head'] is ah and result['B']['head'] is bh


def test_ambiguous_or_unobserved_part_is_not_assigned():
    people=[Detection('person',(10,10,110,410),.9,key=k) for k in ('A','B')]
    head=Detection('head',(30,20,70,70),.9)
    assert associate_body_parts(people,[head])=={'A':{},'B':{}}
    head.observed=False
    assert associate_body_parts(people[:1],[head])=={'A':{}}


def test_crop_boxes_restore_to_each_original_frame_once(frame_factory):
    from types import SimpleNamespace
    import torch
    from app.inference import Detector
    class Localizer:
        def predict(self,images,**kwargs):
            assert kwargs['rect'] is False
            return [SimpleNamespace(orig_shape=im.shape[:2],names={0:'head'},
                boxes=SimpleNamespace(data=torch.tensor([[30,30,50,60,.9,0]]))) for im in images]
    frames=[frame_factory(.1,camera='A',frame_id=11),frame_factory(.2,camera='B',frame_id=22,width=800,height=600)]
    output=[[Detection('person',(100,100,200,400),.9)],[Detection('person',(400,200,600,500),.9)]]
    detector=Detector({'microbatch':4});detector.device='cpu'
    detector.predict_body_parts(Localizer(),{'imgsz':640,'confidence':.3,'nms':False,'active_classes':['head']},frames,output)
    assert output[0][1].xyxy==(106,106,126,136)
    assert output[1][1].xyxy==(406,206,426,236)
    assert output[0][1].last_observed_frame==11 and output[1][1].last_observed_frame==22
    assert output[0][1].last_observed_monotonic==.1 and output[1][1].last_observed_monotonic==.2
