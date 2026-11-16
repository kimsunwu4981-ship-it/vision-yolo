import numpy as np
from app.pose_regions import regions_from_pose


def pose():
    p=np.zeros((17,3),float)
    for i,x,y in [(0,130,145),(1,135,136),(3,150,134),
                  (5,170,165),(6,190,155),(11,230,225),(12,245,215)]:
        p[i]=(x,y,.95)
    return p


def test_pose_regions_follow_landmarks_under_translation():
    p=pose();original={name:box for name,box,c in regions_from_pose(p,600,600)}
    p[:,:2]+=[70,90]
    translated={name:box for name,box,c in regions_from_pose(p,600,600)}
    assert set(original)=={'head','torso'}
    for name,box in original.items():
        assert np.allclose(translated[name],np.array(box)+[70,90,70,90])
    assert original['head'][3] < original['torso'][3]


def test_occluded_landmarks_never_create_fractional_fallback():
    p=pose();p[3,2]=.1
    assert [name for name,box,c in regions_from_pose(p,600,600)]==['torso']
    p[11,2]=.1
    assert regions_from_pose(p,600,600)==[]
    assert regions_from_pose(np.full((17,3),np.nan),600,600)==[]


def test_pose_crop_offset_restored_once(frame_factory):
    from types import SimpleNamespace
    import torch
    from app.types import Detection
    from app.inference import Detector
    class Localizer:
        def predict(self,images,**kwargs):
            return [SimpleNamespace(orig_shape=im.shape[:2],keypoints=SimpleNamespace(data=torch.tensor(pose()[None]))) for im in images]
    frame=frame_factory(.2,frame_id=42,width=800,height=600)
    detections=[[Detection('person',(100,100,400,500),.9)]]
    detector=Detector({'microbatch':4});detector.device='cpu'
    detector.predict_body_parts(Localizer(),{'role':'body_pose','imgsz':640,'confidence':.25,'keypoint_confidence':.5,'nms':False},[frame],detections)
    original={name:box for name,box,c in regions_from_pose(pose(),800,600)}
    for region in detections[0][1:]:
        assert np.allclose(region.xyxy,np.array(original[region.class_name])+[68,68,68,68])
        assert region.localization_basis=='pose_landmarks'
        assert region.last_observed_frame==42 and region.last_observed_monotonic==.2
