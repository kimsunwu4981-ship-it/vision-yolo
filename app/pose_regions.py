"""Anatomical ROIs from visible COCO landmarks, in input-image pixels.

The outlines are landmark-derived estimates, not segmentation or direct part
box predictions. Never substitute a fixed fraction of a person rectangle.
"""
import numpy as np
from app.geometry import validate_box


def regions_from_pose(points, width, height, threshold=.5):
    points = np.asarray(points, dtype=float)
    if points.shape != (17, 3) or not np.isfinite(points).all():
        return []
    visible = points[:, 2] >= threshold
    result = []
    torso_length = None
    if visible[[5,6,11,12]].all():
        torso = points[[5,6,11,12], :2]
        torso_length = float(np.linalg.norm(torso[:2].mean(0)-torso[2:].mean(0)))
        if torso_length >= 12:
            pad = torso_length * .12
            lo, hi = torso.min(0)-pad, torso.max(0)+pad
            result.append(('torso', (*lo, *hi), float(points[[5,6,11,12],2].min())))
    # Need a visible nose, at least one eye and at least one ear. Low-confidence
    # far-side landmarks must not enlarge or move the head outline.
    eyes = [i for i in (1,2) if visible[i]]
    ears = [i for i in (3,4) if visible[i]]
    if visible[0] and eyes and ears:
        nose = points[0,:2]
        eye = points[eyes,:2].mean(0)
        ear = points[ears,:2].mean(0)
        face_axis = nose-eye
        axis_length = float(np.linalg.norm(face_axis))
        scales = [2.2*float(np.linalg.norm(nose-ear)), 2.7*float(np.linalg.norm(eye-ear))]
        if len(ears)==2:
            scales.append(1.35*float(np.linalg.norm(points[3,:2]-points[4,:2])))
        if torso_length is not None:
            scales.append(.5*torso_length)
        diameter = max(scales)
        if diameter >= 10 and axis_length >= 1 and (torso_length is None or diameter < torso_length):
            center = (nose+ear)/2 - .18*diameter*face_axis/axis_length
            result.append(('head', (* (center-diameter/2), * (center+diameter/2)),
                float(points[[0,*eyes,*ears],2].min())))
    valid=[]
    for name, box, confidence in result:
        try:
            box=validate_box(box,width,height)
        except ValueError:
            continue
        valid.append((name,box,confidence))
    return valid
