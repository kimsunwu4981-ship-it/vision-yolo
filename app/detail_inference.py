"""Bounded current-frame detail views; all results remain original-pixel boxes."""
import cv2
import numpy as np
from app.geometry import validate_box,overlap_fraction
from app.types import Detection


def overlap_iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return intersection/max(1.,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)


def valid_ppe_geometry(person,box,kind):
    """Association gate only; never constructs or moves the predicted PPE box."""
    x1,y1,x2,y2=person;w=x2-x1;h=y2-y1;a,b,c,d=box;cx=(a+c)/2;cy=(b+d)/2
    if h<32 or w<10 or min(c-a,d-b)<6 or not x1<=cx<=x2 or not .08<=(c-a)/w<=1.05:return False
    head=kind in {'helmet','bare_head'};low,high=(-.1,.6) if head else (.15,.85)
    return y1+low*h<=cy<=y1+high*h and overlap_fraction(box,(x1,y1-.1*h,x2,y2))>.8 and (d-b)/h<(.6 if head else .8)


def merge_class(items,name,threshold=.65):
    kept=[]
    for item in sorted((d for d in items if d.class_name==name),key=lambda d:-d.confidence):
        if not any(overlap_iou(item.xyxy,other.xyxy)>=threshold for other in kept):kept.append(item)
    return [d for d in items if d.class_name!=name]+kept


def detail_people(detector,model,policy,frames,output):
    config=policy.get('detail',{});jobs=[]
    views=config.get('views',['original','bilateral','flip'])
    if not views or len(views)>3 or set(views)-{'original','bilateral','flip','median','median_flip'}:raise ValueError('invalid_detail_views')
    for index,frame in enumerate(frames):
        if frame.source_width<config.get('min_source_width',1000) or frame.source_width<frame.source_height:continue
        median=cv2.medianBlur(frame.image,3) if {'median','median_flip'}&set(views) else None
        for view in views:
            pixels=frame.image
            if view=='bilateral':pixels=cv2.bilateralFilter(pixels,7,70,70)
            elif view=='flip':pixels=cv2.flip(pixels,1)
            elif view=='median':pixels=median
            elif view=='median_flip':pixels=cv2.flip(median,1)
            jobs.append((index,view,pixels))
    batch_size=max(1,min(2,int(config.get('microbatch',1)),int(detector.config.get('microbatch',1))))
    floor=min(policy['confidence'],config.get('tracking_low_confidence',detector.config.get('tracking_low_confidence',.1)))
    # Group by input size so a mixed batch never silently changes a view's scale.
    sizes=config.get('view_imgsz',{})
    if set(sizes)-set(views) or any(int(v)<640 or int(v)>1920 or int(v)%32 for v in sizes.values()):raise ValueError('invalid_detail_view_size')
    batches=[]
    for size in sorted({int(sizes.get(view,policy['imgsz'])) for _,view,_ in jobs}):
        matching=[j for j in jobs if int(sizes.get(j[1],policy['imgsz']))==size]
        batches.extend((size,matching[start:start+batch_size]) for start in range(0,len(matching),batch_size))
    for size,batch in batches:
        results=model.predict([j[2] for j in batch],device=detector.device,imgsz=size,conf=floor,
            quantize=16 if detector.device.startswith('cuda') and detector.config.get('half',True) else 32,
            rect=True,nms=policy['nms'],max_det=100,verbose=False)
        if len(results)!=len(batch):raise RuntimeError('detail_batch_result_count_mismatch')
        for (index,view,pixels),result in zip(batch,results):
            frame=frames[index]
            if tuple(result.orig_shape)!=(frame.source_height,frame.source_width):raise RuntimeError('detail_frame_shape_mismatch')
            for x1,y1,x2,y2,confidence,cls in result.boxes.data.cpu().numpy():
                if result.names.get(int(cls))!='person' or not np.isfinite(confidence) or confidence<floor:continue
                if view in {'flip','median_flip'}:x1,x2=frame.source_width-x2,frame.source_width-x1
                try:box=validate_box((x1,y1,x2,y2),frame.source_width,frame.source_height)
                except ValueError:continue
                if not config.get('min_person_height',24)<=box[3]-box[1]<=config.get('max_person_height',180):continue
                output[index].append(Detection('person',box,float(confidence),last_observed_frame=frame.frame_id,
                    last_observed_monotonic=frame.received_monotonic,localization_basis='detail_'+view))
    for i in range(len(output)):output[i]=merge_class(output[i],'person')


def context_ppe(detector,model,policy,frames,output):
    config=policy.get('detail',{});jobs=[];limit=max(1,min(16,int(config.get('max_people',12))));candidates=[[] for _ in frames]
    margins=config.get('crop_margins',[config.get('crop_margin',.4)])
    if not margins or len(margins)>3 or any(not np.isfinite(m) or not 0<=m<=2 for m in margins):raise ValueError('invalid_ppe_crop_margins')
    for index,frame in enumerate(frames):
        people=sorted((d for d in output[index] if d.class_name=='person' and d.confidence>=config.get('person_confidence',.25)),key=lambda d:-d.confidence)
        for person in people[:limit]:
            x1,y1,x2,y2=person.xyxy;h=y2-y1
            if not config.get('min_person_height',32)<=h<=config.get('max_person_height',180):continue
            for factor in margins:
                margin=max(8,factor*max(x2-x1,h))
                left=max(0,int(np.floor(x1-margin)));top=max(0,int(np.floor(y1-margin)))
                right=min(frame.source_width,int(np.ceil(x2+margin)));bottom=min(frame.source_height,int(np.ceil(y2+margin)))
                jobs.append((index,left,top,person.xyxy,frame.image[top:bottom,left:right]))
    batch_size=max(1,min(4,int(detector.config.get('microbatch',1))))
    for start in range(0,len(jobs),batch_size):
        batch=jobs[start:start+batch_size]
        results=model.predict([j[4] for j in batch],device=detector.device,imgsz=policy['imgsz'],conf=policy['confidence'],
            quantize=16 if detector.device.startswith('cuda') and detector.config.get('half',True) else 32,
            rect=False,nms=policy['nms'],max_det=16,verbose=False)
        if len(results)!=len(batch):raise RuntimeError('ppe_crop_batch_result_count_mismatch')
        for (index,left,top,owner,crop),result in zip(batch,results):
            frame=frames[index]
            if tuple(result.orig_shape)!=crop.shape[:2]:raise RuntimeError('ppe_crop_shape_mismatch')
            for a,b,c,d,confidence,cls in result.boxes.data.cpu().numpy():
                name=result.names.get(int(cls))
                if name not in policy['active_classes'] or not np.isfinite(confidence):continue
                try:box=validate_box((a+left,b+top,c+left,d+top),frame.source_width,frame.source_height)
                except ValueError:continue
                if not valid_ppe_geometry(owner,box,name):continue
                candidates[index].append(Detection(name,box,float(confidence),last_observed_frame=frame.frame_id,
                    last_observed_monotonic=frame.received_monotonic,localization_basis='person_context_ppe',context_person_xyxy=owner))
    for i in range(len(output)):
        # A worker has one head and one torso. Keep the actual highest-scored
        # predicted box per semantic class/owner; never average/move its edges.
        best={}
        for item in candidates[i]:
            key=(item.class_name,tuple(item.context_person_xyxy))
            if key not in best or item.confidence>best[key].confidence:best[key]=item
        output[i].extend(best.values())
        for name in policy['active_classes']:output[i]=merge_class(output[i],name)
