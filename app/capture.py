from __future__ import annotations
import os
import threading
import time
import uuid
from collections import deque
import av
from app.common import utc_now
from app.types import Frame
from app.orientation import decode_bgr


class LatestSlot:
    def __init__(self):
        self.lock=threading.Lock(); self.frame=None; self.dropped=0
    def put(self,frame):
        with self.lock:
            if self.frame is not None: self.dropped+=1
            self.frame=frame
    def take(self):
        with self.lock:
            frame=self.frame; self.frame=None; return frame
    def clear(self):
        with self.lock: self.frame=None
    def size(self):
        with self.lock: return int(self.frame is not None)


class Capture:
    def __init__(self,config):
        self.config=config; self.slot=LatestSlot(); self.stop_event=threading.Event(); self.thread=None
        self.status="unconfigured"; self.error=None; self.session_id=None
        self.times=deque(maxlen=300); self.total=0; self.last_received=None
        self.shape=None; self.source_fps=None; self.codec=None
        self.seek_request=None; self.seek_lock=threading.Lock()

    @property
    def mode(self): return "LIVE" if self.config.get("source_type")=="rtsp" else ("DEMO" if self.config.get("demo") else "REPLAY")

    def source(self):
        return os.environ.get(self.config.get("url_env",""),"") if self.mode=="LIVE" else self.config.get("path","")

    def start(self):
        if self.thread and self.thread.is_alive(): return
        if not self.source(): self.status="unconfigured";return
        self.stop_event.clear(); self.thread=threading.Thread(target=self.run,name=f"capture-{self.config['id']}",daemon=True); self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=7)
        self.slot.clear(); self.status="stopped"

    def seek(self,seconds):
        if self.mode=="LIVE": raise ValueError("seek_requires_file")
        with self.seek_lock: self.seek_request=max(0.,float(seconds))

    def run(self):
        retry=1.
        while not self.stop_event.is_set():
            self.status="connecting"; self.slot.clear(); self.session_id=uuid.uuid4().hex
            try:
                # FFmpeg network logs can contain credentials. Never enable them.
                av.logging.set_level(av.logging.PANIC)
                options={"rtsp_transport":"tcp","fflags":"nobuffer","flags":"low_delay","reorder_queue_size":"0"} if self.mode=="LIVE" else {}
                with av.open(self.source(),options=options,timeout=(4.,4.)) as container:
                    stream=container.streams.video[0]; stream.thread_type="AUTO"
                    self.source_fps=float(stream.average_rate) if stream.average_rate else None
                    self.codec=stream.codec_context.name; retry=1.; frame_id=0; first_pts=None; base=time.monotonic()
                    while not self.stop_event.is_set():
                        restart=False
                        for decoded in container.decode(stream):
                            if self.stop_event.is_set(): break
                            with self.seek_lock:
                                seek=self.seek_request; self.seek_request=None
                            if seek is not None:
                                container.seek(int(seek/float(stream.time_base)),stream=stream,backward=True)
                                self.session_id=uuid.uuid4().hex; self.slot.clear();frame_id=0;first_pts=None;base=time.monotonic();restart=True;break
                            pts=float(decoded.pts*decoded.time_base) if decoded.pts is not None else None
                            if self.mode!="LIVE" and pts is not None:
                                if first_pts is None: first_pts=pts
                                delay=base+pts-first_pts-time.monotonic()
                                if delay>0 and self.stop_event.wait(delay): break
                            image,transform=decode_bgr(decoded); now=time.monotonic();frame_id+=1
                            height,width=image.shape[:2]
                            frame=Frame(self.config["id"],self.session_id,frame_id,width,height,pts,utc_now(),now,image,preprocess_transform=transform,source_mode=self.mode)
                            self.shape=(width,height);self.last_received=now;self.total+=1;self.times.append(now)
                            self.slot.put(frame);self.status="connected";self.error=None
                        if restart: continue
                        break
                self.slot.clear()
                if self.stop_event.is_set(): break
                if self.mode!="LIVE":
                    if self.config.get("loop",False): continue
                    self.status="ended";break
                raise ConnectionError("stream_ended")
            except Exception as error:
                # Class name only: decoder exception messages may embed RTSP secrets.
                self.error=type(error).__name__;self.status="offline";self.slot.clear()
                if self.stop_event.wait(retry): break
                retry=min(15.,retry*2)

    def metrics(self):
        now=time.monotonic(); stamps=[t for t in list(self.times) if now-t<5]
        fps=(len(stamps)-1)/(stamps[-1]-stamps[0]) if len(stamps)>1 else 0.
        return {"capture_decode_fps":round(fps,2),"captured_frames":self.total,"frame_drops":self.slot.dropped,
                "queue_size":self.slot.size(),"source_size":self.shape,"source_fps":self.source_fps,"codec":self.codec,
                "receive_age_s":None if self.last_received is None else round(now-self.last_received,3)}
