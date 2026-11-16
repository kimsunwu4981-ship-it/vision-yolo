import sqlite3
import time
import pytest
from app.storage import Store
from app.events import EventEngine
from app.types import RuleObservation


def test_locked_database_fails_bounded_then_recovers(tmp_path,frame_factory):
    store=Store(tmp_path,{"min_free_gb":0});store.thread.start()
    blocker=store.connection();blocker.execute('BEGIN IMMEDIATE')
    event=EventEngine().update(frame_factory(0),[RuleObservation('one','ppe_violation',True,confirm_s=0)],'m','c')[0]
    start=time.monotonic();store.save_event(event);store.jobs.join()
    assert time.monotonic()-start<2 and store.failed==1
    assert store.error=='OperationalError'
    blocker.rollback();blocker.close()
    store.save_event(event);store.jobs.join();assert store.list_events()[0]['id']==event['id']
    store.close()


def test_startup_db_failure_does_not_abort_store(tmp_path,monkeypatch):
    def fail(self):raise sqlite3.OperationalError('test database unavailable')
    monkeypatch.setattr(Store,'initialize',fail)
    store=Store(tmp_path,{})
    assert store.error=='OperationalError'


def test_token_auth_and_origin_restrictions(tmp_path,monkeypatch):
    from app.api import create_app
    from app.config import load_config
    from app.runtime import Runtime
    from fastapi.testclient import TestClient
    monkeypatch.setenv('SAFETY_API_TOKEN','integration-test-token-123456789')
    monkeypatch.setenv('SAFETY_HOST','0.0.0.0')
    cfg=load_config();runtime=Runtime(cfg,tmp_path);monkeypatch.setattr(runtime.detector,'load',lambda:None)
    with TestClient(create_app(cfg,runtime)) as client:
        assert client.get('/api/system').status_code==401
        assert client.post('/api/auth',json={'token':'wrong'}).status_code==401
        response=client.post('/api/auth',json={'token':'integration-test-token-123456789'})
        assert response.status_code==200 and 'HttpOnly' in response.headers['set-cookie']
        assert client.get('/api/system').status_code==200
        assert client.post('/api/cameras/cam1/start',headers={'origin':'https://foreign.example'}).status_code==403


def test_second_worker_cannot_own_pipeline(tmp_path,monkeypatch):
    from app.runtime import Runtime
    from app.config import load_config
    monkeypatch.setattr('app.runtime.ROOT',tmp_path)
    first=Runtime(load_config(),tmp_path/'a');second=Runtime(load_config(),tmp_path/'b')
    monkeypatch.setattr(first.detector,'load',lambda:None)
    first.start()
    try:
        with pytest.raises(RuntimeError,match='one_worker'):second.start()
        first.start()  # Same owner is idempotent, e.g. mounted API lifespan.
    finally:first.close()
