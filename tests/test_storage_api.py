import time
from pathlib import Path
import cv2
import pytest
from fastapi.testclient import TestClient
from app.storage import Store
from app.config import load_config,CameraConfig,ZoneConfig
from app.events import EventEngine
from app.types import RuleObservation


def test_evidence_ack_restart_and_retention(tmp_path,frame_factory):
    store=Store(tmp_path,{"min_free_gb":0,"pre_seconds":1,"post_seconds":.1,"retention_days":14,"max_gb":1})
    store.thread.start();frame=frame_factory(time.monotonic());jpeg=cv2.imencode('.jpg',frame.image)[1].tobytes()
    event=EventEngine().update(frame,[RuleObservation("risk","zone_intrusion",True,confirm_s=0)],"model","revision")[0]
    store.add_frame(frame,jpeg);store.save_event(event,frame,jpeg)
    store.jobs.join();assert store.acknowledge(event["id"])
    assert store.list_events()[0]["status"]=="active" and store.list_events()[0]["acknowledged"]
    assert store.evidence_path(event["id"]+"-image").is_file()
    time.sleep(.3);store.close()
    assert store.evidence_path(event["id"]+"-clip").is_file()
    restarted=Store(tmp_path,{"min_free_gb":0});assert restarted.list_events()[0]["observation_state"]=="unknown_after_restart"
    assert restarted.evidence_path("../../outside") is None
    outside=tmp_path/"keep.txt";outside.write_text("preserve")
    restarted.config["retention_days"]=-1;restarted.retention();assert outside.exists()

def test_disk_full_and_queue_bounded(tmp_path):
    store=Store(tmp_path,{"min_free_gb":1e10})
    store.write_evidence("x","image",b"jpeg");assert store.error=="insufficient_disk_space"
    for i in range(200):store.submit("event",{})
    assert store.jobs.qsize()==128 and store.dropped==72

def test_validation_rejects_secrets_and_invalid_roi():
    with pytest.raises(ValueError):CameraConfig(id="cam1",name="c",source_type="file",path="rtsp://user:secret@host/")
    with pytest.raises(ValueError):CameraConfig(id="../bad",name="c")
    with pytest.raises(ValueError):ZoneConfig(id="x",name="z",points=[[0,0],[1,1],[0,1],[1,0]])

def test_api_dashboard_health_auth_and_validation(tmp_path,monkeypatch):
    from app.api import create_app
    from app.runtime import Runtime
    cfg=load_config();runtime=Runtime(cfg,tmp_path)
    # API contract test deliberately avoids claiming model/inference validation.
    monkeypatch.setattr(runtime.detector,"load",lambda:None)
    monkeypatch.setattr(runtime,"set_zones",lambda cid,zones:"testrevision")
    with TestClient(create_app(cfg,runtime)) as client:
        assert client.get('/').status_code==200
        assert client.get('/health/live').json()["alive"]
        assert client.get('/health/ready').status_code==503
        data=client.get('/api/cameras').json();assert len(data)==4 and 'url_env' not in str(data)
        assert client.put('/api/cameras/cam1/zones',json=[{"id":"a","name":"a","points":[[0,0],[1,0],[1,1]]}]).status_code==200
        assert client.put('/api/cameras/cam1/zones',json=[{"id":"a","name":"a","points":[[0,0],[2,0],[1,1]]}]).status_code==422
        assert client.post('/api/events/missing/ack').status_code==404
        assert client.get('/api/evidence/missing').status_code==404
        assert client.post('/api/cameras/cam1/stop',headers={"Origin":"https://hostile.example"}).status_code==403
        with client.websocket_connect('/ws/events') as ws:assert 'state' in ws.receive_json()

def test_lan_requires_token(tmp_path,monkeypatch):
    from app.api import create_app
    from app.runtime import Runtime
    monkeypatch.setenv("SAFETY_HOST","0.0.0.0");monkeypatch.delenv("SAFETY_API_TOKEN",raising=False)
    cfg=load_config()
    with pytest.raises(RuntimeError):create_app(cfg,Runtime(cfg,tmp_path))
