"""One Ultralytics BYTETracker per camera/session/class; no second tracker."""
from types import SimpleNamespace
import numpy as np
from app.types import Detection


class TrackerBank:
    def __init__(self, ttl_s=1.0):
        self.ttl_s=ttl_s
        self.banks={}
        self.observed={}
        self.args=SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,
                                  track_buffer=30,match_thresh=.8,fuse_score=True)

    def reset_camera(self,camera):
        self.banks={k:v for k,v in self.banks.items() if k[0]!=camera}

    def update(self,frame,detections):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.engine.results import Boxes
        # A reconnect discards every previous session, including Kalman state.
        self.banks={k:v for k,v in self.banks.items() if k[0]!=frame.camera_id or k[1]==frame.session_id}
        output=[d for d in detections if d.class_name not in {"person","forklift"}]
        for name in ("person","forklift"):
            key=(frame.camera_id,frame.session_id,name)
            prior=self.banks.get(key)
            if prior is None:
                prior=(BYTETracker(self.args),frame.received_monotonic)
            tracker,last=prior
            dt=max(.001,frame.received_monotonic-last)
            tracker.max_frames_lost=max(1,round(self.ttl_s/dt))
            # Wall-time expiry is authoritative even with dropped frames or variable FPS.
            for pool in ("tracked_stracks","lost_stracks"):
                setattr(tracker,pool,[track for track in getattr(tracker,pool)
                    if frame.received_monotonic-self.observed.get((*key,track.track_id),-1e10)<=self.ttl_s])
            subset=[d for d in detections if d.class_name==name]
            raw=np.array([[*d.xyxy,d.confidence,0] for d in subset],np.float32).reshape(-1,6)
            rows=tracker.update(Boxes(raw,(frame.source_height,frame.source_width)),frame.image)
            for candidate in tracker.tracked_stracks:
                if candidate.frame_id==tracker.frame_id:
                    self.observed[(*key,candidate.track_id)]=frame.received_monotonic
            assigned=set()
            for row in rows:
                # BYTETracker returns only currently observed, activated tracks.
                idx=int(row[-1])
                if not 0<=idx<len(subset): continue
                assigned.add(idx)
                detection=subset[idx]
                detection.track_id=int(row[4]); detection.last_observed_frame=frame.frame_id
                detection.last_observed_monotonic=frame.received_monotonic
                detection.key=f"{frame.camera_id}:{frame.session_id}:{name}:{detection.track_id}"
                self.observed[(*key,detection.track_id)]=frame.received_monotonic
                output.append(detection)
            output.extend(d for i,d in enumerate(subset) if i not in assigned)
            self.banks[key]=(tracker,frame.received_monotonic)
        self.observed={k:v for k,v in self.observed.items() if frame.received_monotonic-v<30 and (k[0]!=frame.camera_id or k[1]==frame.session_id)}
        return output
