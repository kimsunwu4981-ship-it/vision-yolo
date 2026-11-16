"""Fit ground-plane homography from measured correspondences; requires field validation."""
import argparse
import cv2
import numpy as np
from app.common import read_json,write_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('points');p.add_argument('--output',required=True);a=p.parse_args();data=read_json(a.points)
    pixels=np.asarray(data['pixel_points'],np.float64);world=np.asarray(data['ground_metres'],np.float64)
    if pixels.shape!=world.shape or pixels.ndim!=2 or pixels.shape[1]!=2 or len(pixels)<4 or not np.isfinite(pixels).all() or not np.isfinite(world).all():raise ValueError('at_least_four_finite_correspondences_required')
    h,mask=cv2.findHomography(pixels,world,method=0)
    if h is None or abs(np.linalg.det(h))<1e-10:raise ValueError('degenerate_calibration')
    estimated=cv2.perspectiveTransform(pixels.reshape(-1,1,2),h).reshape(-1,2)
    report={'homography':h.tolist(),'floor_points':pixels.tolist(),'ground_metres':world.tolist(),'unit':'metres',
        'source_size':data['source_size'],'fit_rmse_m':float(np.sqrt(((estimated-world)**2).sum(axis=1).mean())),
        'validated':False,'threshold_m':2.,'instruction':'Verify independent measured checkpoints in the actual camera view before setting validated=true. A four-point fit is not independent validation.'}
    write_json(a.output,report)
