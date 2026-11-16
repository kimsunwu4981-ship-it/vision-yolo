"""Coordinate contracts. Ultralytics original xyxy must not be inverted twice."""
import math
import numpy as np
import cv2


def validate_box(box, width, height):
    a = np.asarray(box, dtype=float)
    if a.shape != (4,) or not np.isfinite(a).all():
        raise ValueError("nonfinite_or_bad_shape")
    x1,y1,x2,y2 = a
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("box_outside_original_frame")
    return tuple(map(float, a))


def transform_boxes(boxes, matrix):
    """Transform all four corners by a homography (crop/resize/rotation included)."""
    output=[]
    for x1,y1,x2,y2 in boxes:
        p=np.array([[x1,y1,1],[x2,y1,1],[x2,y2,1],[x1,y2,1]],dtype=float).T
        q=np.asarray(matrix,dtype=float) @ p
        if (np.abs(q[2]) < 1e-10).any(): raise ValueError("singular_projection")
        q=q[:2]/q[2]
        output.append([q[0].min(),q[1].min(),q[0].max(),q[1].max()])
    return np.asarray(output,dtype=float).reshape(-1,4)


def letterbox_matrix(width,height,target_width,target_height):
    scale=min(target_width/width,target_height/height)
    return np.array([[scale,0,(target_width-width*scale)/2],[0,scale,(target_height-height*scale)/2],[0,0,1]],float)


def foot(box):
    x1,y1,x2,y2=box
    return ((x1+x2)/2,y2)


def overlap_fraction(a,b):
    ax,ay,bx,by=a; cx,cy,dx,dy=b
    intersection=max(0,min(bx,dx)-max(ax,cx))*max(0,min(by,dy)-max(ay,cy))
    return intersection/max(1.,(bx-ax)*(by-ay))


def polygon_distance(point, polygon):
    return float(cv2.pointPolygonTest(np.asarray(polygon,dtype=np.float32),tuple(map(float,point)),True))


def validate_polygon(points):
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=2 or not 3<=len(p)<=64 or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError("polygon_requires_3_to_64_normalized_points")
    if abs(cv2.contourArea(p.astype(np.float32)))<1e-5: raise ValueError("polygon_area_too_small")
    # Reject self intersections, including collinear overlaps.
    def orient(a,b,c): return float((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
    for i in range(len(p)):
        a,b=p[i],p[(i+1)%len(p)]
        if np.linalg.norm(a-b)<1e-6: raise ValueError("duplicate_polygon_point")
        for j in range(i+1,len(p)):
            if j==i+1 or (i==0 and j==len(p)-1): continue
            c,d=p[j],p[(j+1)%len(p)]
            if orient(a,b,c)*orient(a,b,d)<=0 and orient(c,d,a)*orient(c,d,b)<=0:
                if np.all(np.maximum(np.minimum(a,b),np.minimum(c,d)) <= np.minimum(np.maximum(a,b),np.maximum(c,d))):
                    raise ValueError("self_intersecting_polygon")
    return p.tolist()


def projected_distance(a,b,homography):
    h=np.asarray(homography,float)
    if h.shape!=(3,3) or not np.isfinite(h).all() or abs(np.linalg.det(h))<1e-10: raise ValueError("invalid_homography")
    points=[]
    for x,y in [a,b]:
        q=h@np.array([x,y,1.]);
        if abs(q[2])<1e-9: raise ValueError("invalid_projection")
        points.append(q[:2]/q[2])
    return float(np.linalg.norm(points[0]-points[1]))
