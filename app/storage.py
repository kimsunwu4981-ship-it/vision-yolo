from __future__ import annotations
import json
import queue
import shutil
import sqlite3
import threading
import time
import uuid
from collections import deque
from fractions import Fraction
from pathlib import Path
import av
import cv2
import numpy as np
from app.common import utc_now


class Store:
    def __init__(self,root,config):
        self.root=Path(root).resolve(); self.root.mkdir(parents=True,exist_ok=True)
        self.evidence=(self.root/"evidence").resolve();self.evidence.mkdir(exist_ok=True)
        self.marker=self.evidence/".industrial-safety-owned";self.marker.touch(exist_ok=True)
        self.db=self.root/"events.sqlite3";self.config=config
        self.jobs=queue.Queue(maxsize=128);self.stop=threading.Event();self.error=None;self.dropped=0;self.failed=0
        self.buffers={};self.pending={};self.buffer_lock=threading.Lock();self.buffer_bytes=0
        self.thread=threading.Thread(target=self.run,name="storage",daemon=True)
        try:self.initialize()
        except (sqlite3.Error,OSError) as error:self.error=type(error).__name__

    def connection(self):
        con=sqlite3.connect(self.db,timeout=.3);con.row_factory=sqlite3.Row;return con
    def initialize(self):
        with self.connection() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript("CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, camera_id TEXT, type TEXT, status TEXT, started_at TEXT, acknowledged INTEGER DEFAULT 0, body TEXT); CREATE INDEX IF NOT EXISTS event_time ON events(started_at); CREATE TABLE IF NOT EXISTS evidence(id TEXT PRIMARY KEY,path TEXT,kind TEXT,created_at TEXT); ")
            # Restart is not proof that prior hazards became safe.
            for row in con.execute("SELECT id,body FROM events WHERE status != 'resolved'").fetchall():
                body=json.loads(row["body"]);body["observation_state"]="unknown_after_restart"
                con.execute("UPDATE events SET body=? WHERE id=?",(json.dumps(body),row["id"]))

    def submit(self,kind,payload):
        try:self.jobs.put_nowait((kind,payload));return True
        except queue.Full:self.dropped+=1;self.error="storage_queue_full";return False

    def add_frame(self,frame,jpeg):
        with self.buffer_lock:
            key=(frame.camera_id,frame.session_id)
            # Sessions are isolated; discard previous pre-event buffers.
            for old in list(self.buffers):
                if old[0]==frame.camera_id and old!=key:
                    self.buffer_bytes-=sum(len(item[1]) for item in self.buffers.pop(old))
            buf=self.buffers.setdefault(key,deque())
            buf.append((frame.received_monotonic,jpeg));self.buffer_bytes+=len(jpeg)
            while buf and (frame.received_monotonic-buf[0][0]>self.config.get("pre_seconds",3) or len(buf)>90):
                self.buffer_bytes-=len(buf.popleft()[1])
            cap=self.config.get("buffer_mb",64)*1024**2
            while self.buffer_bytes>cap:
                nonempty=[b for b in self.buffers.values() if b]
                if not nonempty:break
                oldest=min(nonempty,key=lambda b:b[0][0]);self.buffer_bytes-=len(oldest.popleft()[1])
            for event_id,clip in list(self.pending.items()):
                if clip["key"]==key and frame.received_monotonic>clip["frames"][-1][0]:
                    if clip["bytes"]+len(jpeg)<=self.config.get("clip_mb",16)*1024**2:
                        clip["frames"].append((frame.received_monotonic,jpeg));clip["bytes"]+=len(jpeg)
                if frame.received_monotonic>=clip["deadline"]:
                    self.submit("clip",(event_id,clip["frames"]));del self.pending[event_id]

    def save_event(self,event,frame=None,jpeg=None):
        self.submit("event",event.copy())
        if event["status"]=="active" and frame is not None and jpeg is not None:
            with self.buffer_lock:
                if event["id"] not in self.pending and not self.has_evidence(event["id"]+"-image"):
                    self.submit("image",(event["id"],jpeg))
                    if len(self.pending)<self.config.get("max_pending_clips",8):
                        frames=list(self.buffers.get((frame.camera_id,frame.session_id),[])) or [(frame.received_monotonic,jpeg)]
                        self.pending[event["id"]]={"key":(frame.camera_id,frame.session_id),"frames":frames,
                            "bytes":sum(len(x[1]) for x in frames),"deadline":frame.received_monotonic+self.config.get("post_seconds",3)}

    def has_evidence(self,evidence_id):
        try:
            with self.connection() as con:return con.execute("SELECT 1 FROM evidence WHERE id=?",(evidence_id,)).fetchone() is not None
        except sqlite3.Error:return False

    def list_events(self,camera=None,kind=None,status=None,limit=100):
        clauses=[];params=[]
        for key,value in [("camera_id",camera),("type",kind),("status",status)]:
            if value:clauses.append(key+"=?");params.append(value)
        sql="SELECT * FROM events"+(" WHERE "+" AND ".join(clauses) if clauses else "")+" ORDER BY started_at DESC LIMIT ?"
        params.append(min(500,max(1,limit)))
        with self.connection() as con:
            output=[]
            for row in con.execute(sql,params):
                body=json.loads(row["body"]);body["acknowledged"]=bool(row["acknowledged"])
                for suffix in ["image","clip"]:
                    if con.execute("SELECT 1 FROM evidence WHERE id=?",(row["id"]+"-"+suffix,)).fetchone():body[suffix+"_id"]=row["id"]+"-"+suffix
                output.append(body)
            return output

    def acknowledge(self,event_id):
        with self.connection() as con:
            result=con.execute("UPDATE events SET acknowledged=1 WHERE id=?",(event_id,));return result.rowcount>0

    def evidence_path(self,evidence_id):
        with self.connection() as con: row=con.execute("SELECT path FROM evidence WHERE id=?",(evidence_id,)).fetchone()
        if not row:return None
        path=(self.evidence/row["path"]).resolve()
        return path if path.is_relative_to(self.evidence) and path.is_file() else None

    def disk_ok(self):
        return shutil.disk_usage(self.evidence).free>=self.config.get("min_free_gb",2)*1024**3

    def write_evidence(self,event_id,kind,payload):
        if not self.disk_ok():self.error="insufficient_disk_space";return
        identifier=event_id+"-"+kind;path=self.evidence/(identifier+(".jpg" if kind=="image" else ".mp4"))
        if kind=="image": path.write_bytes(payload)
        else:
            if not payload:return
            first=cv2.imdecode(np.frombuffer(payload[0][1],np.uint8),cv2.IMREAD_COLOR)
            if first is None:return
            h,w=first.shape[:2]; w-=w%2;h-=h%2
            with av.open(str(path),"w") as container:
                stream=container.add_stream("libx264",rate=15);stream.width=w;stream.height=h;stream.pix_fmt="yuv420p"
                stream.options={"crf":"25","preset":"veryfast","tune":"zerolatency"};stream.time_base=Fraction(1,1000)
                stream.codec_context.time_base=Fraction(1,1000);stream.codec_context.max_b_frames=0
                start=payload[0][0];last=-1
                for timestamp,jpeg in payload:
                    image=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
                    if image is None:continue
                    frame=av.VideoFrame.from_ndarray(cv2.resize(image,(w,h)),format="bgr24")
                    frame.pts=max(last+1,round((timestamp-start)*1000));last=frame.pts;frame.time_base=Fraction(1,1000)
                    for packet in stream.encode(frame):container.mux(packet)
                for packet in stream.encode():container.mux(packet)
        with self.connection() as con:
            con.execute("INSERT OR REPLACE INTO evidence VALUES(?,?,?,?)",(identifier,path.name,kind,utc_now()))

    def retention(self):
        # Only files named by this application's evidence table, under the marked root.
        if not self.marker.is_file():raise RuntimeError("owned_evidence_marker_missing")
        with self.connection() as con:
            rows=con.execute("SELECT id,path FROM evidence ORDER BY created_at").fetchall()
            files=[]
            for row in rows:
                path=(self.evidence/row["path"]).resolve()
                if path.is_relative_to(self.evidence) and path.is_file():files.append((row["id"],path,path.stat()))
            total=sum(st.st_size for _,_,st in files); maximum=self.config.get("max_gb",10)*1024**3
            for identifier,path,stat in files:
                if time.time()-stat.st_mtime>self.config.get("retention_days",14)*86400 or total>maximum or not self.disk_ok():
                    path.unlink();total-=stat.st_size;con.execute("DELETE FROM evidence WHERE id=?",(identifier,))

    def run(self):
        last_cleanup=0
        while not self.stop.is_set() or not self.jobs.empty():
            try:kind,payload=self.jobs.get(timeout=.2)
            except queue.Empty:
                # Finish post-roll even when a camera stopped supplying frames.
                with self.buffer_lock:
                    for event_id,clip in list(self.pending.items()):
                        if time.monotonic()>clip["deadline"]:
                            self.submit("clip",(event_id,clip["frames"]));del self.pending[event_id]
                continue
            try:
                if kind=="event":
                    with self.connection() as con:
                        con.execute("INSERT INTO events(id,camera_id,type,status,started_at,body) VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,body=excluded.body",
                            (payload["id"],payload["camera_id"],payload["type"],payload["status"],payload["started_at"],json.dumps(payload)))
                elif kind in {"image","clip"}:self.write_evidence(payload[0],kind,payload[1])
                if time.monotonic()-last_cleanup>60:self.retention();last_cleanup=time.monotonic()
            except Exception as error:self.error=type(error).__name__;self.failed+=1
            finally:self.jobs.task_done()

    def close(self):
        with self.buffer_lock:
            for event_id,clip in self.pending.items(): self.submit("clip",(event_id,clip["frames"]))
            self.pending.clear()
        self.stop.set()
        if self.thread.is_alive():self.thread.join(timeout=10)
