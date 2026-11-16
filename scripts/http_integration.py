"""Real registered model + generated video + actual HTTP/MJPEG/WebSocket integration."""
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import httpx
import av
from app.common import ROOT,write_json,utc_now


def run():
    from app.common import read_json
    import psutil
    owned=read_json(ROOT/'.runtime/service.json',{})
    if owned and psutil.pid_exists(owned.get('pid',-1)):
        raise RuntimeError('Stop the existing project service before this integration test')
    directory=ROOT/"outputs/http-integration";directory.mkdir(parents=True,exist_ok=True)
    environment=os.environ.copy();environment["SAFETY_CAMERAS"]=str(ROOT/"configs/cameras.demo.local.yaml");environment["SAFETY_PORT"]="8766"
    environment["SAFETY_HOST"]="127.0.0.1";environment.pop("SAFETY_API_TOKEN",None)
    environment["SAFETY_OUTPUT_DIR"]="outputs/http-integration/runtime"
    # Back up only this project's test ROI; restore even if an assertion fails.
    zone_path=ROOT/"configs/zones/cam1.json";prior_zone=zone_path.read_bytes() if zone_path.exists() else None
    result={"started_at":utc_now(),"fixture":"generated still-photo videos, not real surveillance video","checks":{}}
    log=open(directory/"server.log","w",encoding="utf-8")
    process=subprocess.Popen([sys.executable,"-m","app.main"],cwd=ROOT,env=environment,stdout=log,stderr=log)
    client=httpx.Client(base_url="http://127.0.0.1:8766",timeout=8,trust_env=False)
    try:
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            try:
                state=client.get('/api/system').json()
                if state['model']['ready'] and all(c['inferred_frames']>3 for c in state['cameras']):break
            except (httpx.HTTPError,ValueError):pass
            if process.poll() is not None:raise RuntimeError('server_exited')
            time.sleep(.3)
        assert state['model']['ready'] and state['model']['actual_device'].startswith('cuda')
        result['checks']['actual_cuda_model']=True
        assert 'forklift' in state['model']['missing_classes'];assert client.get('/health/ready').status_code==503
        result['checks']['partial_model_not_operational_ready']=True
        for cid in ['cam1','cam2','cam3','cam4']:
            with client.stream('GET',f'/api/cameras/{cid}/stream.mjpg') as response:
                assert response.status_code==200
                data=b''
                for chunk in response.iter_bytes():
                    data+=chunk
                    begin=data.find(b'\xff\xd8');end=data.find(b'\xff\xd9',begin+2)
                    if begin>=0 and end>begin:
                        (directory/(cid+'.jpg')).write_bytes(data[begin:end+2]);break
                    assert len(data)<5*1024**2
                assert b'X-Frame-Key:' in data and cid.encode() in data
        result['checks']['four_mjpeg_actual_model_frames']=True
        before={c['id']:c['inferred_frames'] for c in client.get('/api/cameras').json()}
        assert client.post('/api/cameras/cam4/stop').status_code==200
        time.sleep(1.5)
        stopped=client.get('/api/cameras').json();assert stopped[3]['status']=='stopped' and not stopped[3]['analysis_live']
        result['stop_isolation_measurement']={'before':before,'after':{c['id']:c['inferred_frames'] for c in stopped},'observation_age_s':{c['id']:c['last_observation_age_s'] for c in stopped}}
        assert all(c['inferred_frames']>before[c['id']] for c in stopped[:3])
        result['checks']['stopped_camera_isolation']=True
        old_session=stopped[3]['session_id'];client.post('/api/cameras/cam4/start');time.sleep(1)
        assert client.get('/api/cameras').json()[3]['session_id']!=old_session
        result['checks']['restart_new_session']=True
        prior_ids={e['id'] for e in client.get('/api/events').json()}
        zone=[dict(id='integration-zone',name='통합 시험 구역',points=[[.01,.01],[.99,.01],[.99,.99],[.01,.99]],enter_s=.1,exit_s=.1,margin=0)]
        assert client.put('/api/cameras/cam1/zones',json=zone).status_code==200
        assert client.get('/api/cameras/cam1/zones').json()['zones'][0]['id']=='integration-zone'
        result['checks']['roi_save_reload']=True
        deadline=time.monotonic()+15;found=None
        while time.monotonic()<deadline:
            events=client.get('/api/events',params={'kind':'zone_intrusion','camera':'cam1'}).json()
            found=next((e for e in events if e['id'] not in prior_ids and e.get('source_mode')=='DEMO' and e.get('image_id') and e.get('clip_id')),None)
            if found:break
            time.sleep(.3)
        assert found,'No actual-model zone event with evidence'
        assert client.post('/api/events/'+found['id']+'/ack').json()['acknowledged']
        updated=next(e for e in client.get('/api/events').json() if e['id']==found['id'])
        assert updated['acknowledged'] and updated['status']==found['status']
        result['checks']['actual_model_event_ack']=True
        image=client.get('/api/evidence/'+found['image_id']);assert image.status_code==200 and image.content.startswith(b'\xff\xd8')
        clip=client.get('/api/evidence/'+found['clip_id']);assert clip.status_code==200
        path=directory/'event.mp4';path.write_bytes(clip.content)
        with av.open(str(path)) as container:assert sum(1 for _ in container.decode(video=0))>1
        result['checks']['evidence_http_and_video_decode']=True
        async def websocket_checks():
            import websockets
            for _ in range(2):
                async with websockets.connect('ws://127.0.0.1:8766/ws/events',origin='http://127.0.0.1:8766',max_queue=1) as ws:
                    payload=json.loads(await asyncio.wait_for(ws.recv(),5));assert len(payload['state']['cameras'])==4
        asyncio.run(websocket_checks());result['checks']['websocket_reconnect']=True
        result['final_state']=client.get('/api/system').json();result['passed']=True
    except Exception as error:
        result['passed']=False;result['error']=repr(error)
        try:result['failure_state']=client.get('/api/system').json()
        except Exception:pass
        raise
    finally:
        client.close()
        subprocess.run([sys.executable,'-m','scripts.stop_server'],cwd=ROOT,capture_output=True,timeout=40)
        if process.poll() is None:process.terminate();process.wait(timeout=10)
        log.close()
        if prior_zone is None:zone_path.unlink(missing_ok=True)
        else:zone_path.write_bytes(prior_zone)
        write_json(directory/'result.json',result)
    print(json.dumps(result['checks']))

if __name__=='__main__':run()
