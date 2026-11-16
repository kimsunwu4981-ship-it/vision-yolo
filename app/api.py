from __future__ import annotations
import asyncio
import hmac
import os
import secrets
import time
from contextlib import asynccontextmanager
from urllib.parse import urlsplit
from fastapi import FastAPI,Request,HTTPException,WebSocket,WebSocketDisconnect
from fastapi.responses import FileResponse,JSONResponse,StreamingResponse
from pydantic import BaseModel,Field
from app.common import ROOT,read_json
from app.config import CameraConfig,ZoneConfig,load_config
from app.runtime import Runtime


def create_app(config=None,runtime=None):
    config=config or load_config()
    output=(ROOT/os.environ.get("SAFETY_OUTPUT_DIR","outputs/runtime")).resolve()
    if not output.is_relative_to(ROOT):raise ValueError("runtime_output_must_be_inside_project")
    runtime=runtime or Runtime(config,output)
    sessions={};token=os.environ.get("SAFETY_API_TOKEN","");host=os.environ.get("SAFETY_HOST","127.0.0.1")
    if host not in {"127.0.0.1","localhost","::1"} and len(token)<24:raise RuntimeError("LAN_requires_SAFETY_API_TOKEN_24_characters")
    def authorized(cookie):return not token or sessions.get(cookie,0)>time.monotonic()
    @asynccontextmanager
    async def lifespan(app):
        runtime.start()
        try:yield
        finally:await asyncio.to_thread(runtime.close)
    app=FastAPI(title="Industrial Safety YOLO",lifespan=lifespan)
    app.state.runtime=runtime
    @app.middleware("http")
    async def security(request,call_next):
        origin=request.headers.get("origin")
        if origin and request.method not in {"GET","HEAD","OPTIONS"}:
            if urlsplit(origin).netloc!=request.headers.get("host"):return JSONResponse({"detail":"origin_not_allowed"},403)
        if request.url.path.startswith("/api/") and request.url.path!="/api/auth" and not authorized(request.cookies.get("safety_session")):
            return JSONResponse({"detail":"authentication_required"},401)
        response=await call_next(request)
        response.headers["X-Content-Type-Options"]="nosniff";response.headers["Referrer-Policy"]="no-referrer"
        response.headers["Cache-Control"]="no-store"
        return response
    @app.get("/")
    async def index():return FileResponse(ROOT/"frontend/dashboard.html")
    @app.get("/health/live")
    async def live():return {"alive":True}
    @app.get("/health/ready")
    async def ready():
        ok=runtime.detector.ready and not runtime.detector.degraded and all(c["analysis_live"] for c in runtime.camera_status()) and len(runtime.cameras)==4
        return JSONResponse({"ready":ok,"software_running":True,"field_acceptance":False,"model":runtime.detector.status()},200 if ok else 503)
    @app.post("/api/auth")
    async def auth(request:Request):
        value=await request.json()
        if not token or not hmac.compare_digest(str(value.get("token","")),token):raise HTTPException(401,"invalid_token")
        now=time.monotonic()
        for key in list(sessions):
            if sessions[key]<now:del sessions[key]
        if len(sessions)>32:raise HTTPException(429,"session_limit")
        sid=secrets.token_urlsafe(32);sessions[sid]=now+43200
        response=JSONResponse({"authenticated":True});response.set_cookie("safety_session",sid,httponly=True,samesite="strict",max_age=43200,secure=request.url.scheme=="https")
        return response
    @app.get("/api/system")
    async def system():return runtime.snapshot()
    @app.get("/api/models")
    async def models():return {"runtime":runtime.detector.status(),"registry":read_json(ROOT/"models/model_registry.json",{})}
    @app.get("/api/metrics")
    async def metrics():return runtime.snapshot()
    @app.get("/api/cameras")
    async def cameras():return runtime.camera_status()
    @app.post("/api/cameras",status_code=201)
    async def add_camera(camera:CameraConfig):
        try:runtime.add_camera(camera.model_dump())
        except ValueError as error:raise HTTPException(400,str(error))
        return {"id":camera.id}
    def camera(cid):
        if cid not in runtime.cameras:raise HTTPException(404,"camera_not_found")
        return runtime.cameras[cid]
    @app.post("/api/cameras/{cid}/start")
    async def start(cid:str):camera(cid).start();return {"status":camera(cid).status}
    @app.post("/api/cameras/{cid}/stop")
    async def stop(cid:str):await asyncio.to_thread(camera(cid).stop);return {"status":"stopped"}
    class Seek(BaseModel):seconds:float=Field(ge=0,le=1e8)
    @app.post("/api/cameras/{cid}/seek")
    async def seek(cid:str,body:Seek):
        try:camera(cid).seek(body.seconds)
        except ValueError as error:raise HTTPException(400,str(error))
        return {"requested_seconds":body.seconds,"new_session":True}
    @app.get("/api/cameras/{cid}/zones")
    async def zones(cid:str):camera(cid);return {"zones":runtime.zones[cid],"revision":runtime.config_revision}
    @app.put("/api/cameras/{cid}/zones")
    async def save_zones(cid:str,body:list[ZoneConfig]):
        camera(cid)
        if len(body)>20 or len({z.id for z in body})!=len(body):raise HTTPException(400,"unique_zone_ids_max_20")
        rev=runtime.set_zones(cid,[z.model_dump() for z in body]);return {"saved":len(body),"revision":rev}
    @app.get("/api/cameras/{cid}/stream.mjpg")
    async def stream(cid:str,request:Request):
        camera(cid)
        async def frames():
            previous=None
            while not await request.is_disconnected():
                key,jpeg=runtime.preview(cid)
                if key!=previous:
                    previous=key
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "+str(len(jpeg)).encode()+b"\r\nX-Frame-Key: "+str(key).encode()+b"\r\n\r\n"+jpeg+b"\r\n"
                await asyncio.sleep(1/config["runtime"]["preview_fps"])
        return StreamingResponse(frames(),media_type="multipart/x-mixed-replace; boundary=frame",headers={"X-Accel-Buffering":"no"})
    @app.get("/api/events")
    async def events(camera:str|None=None,kind:str|None=None,status:str|None=None,limit:int=100):
        try:return await asyncio.to_thread(runtime.store.list_events,camera,kind,status,limit)
        except Exception:raise HTTPException(503,"event_storage_unavailable")
    @app.post("/api/events/{event_id}/ack")
    async def acknowledge(event_id:str):
        if not await asyncio.to_thread(runtime.store.acknowledge,event_id):raise HTTPException(404,"event_not_found")
        return {"acknowledged":True,"hazard_resolution_unchanged":True}
    @app.get("/api/evidence/{evidence_id}")
    async def evidence(evidence_id:str):
        path=await asyncio.to_thread(runtime.store.evidence_path,evidence_id)
        if not path:raise HTTPException(404,"evidence_missing_or_expired")
        return FileResponse(path)
    @app.websocket("/ws/events")
    async def websocket(ws:WebSocket):
        origin=ws.headers.get("origin")
        if not authorized(ws.cookies.get("safety_session")) or (origin and urlsplit(origin).netloc!=ws.headers.get("host")):
            await ws.close(code=1008);return
        await ws.accept()
        try:
            while True:
                # One current snapshot; slow clients never accumulate a frame/event queue.
                try:events=await asyncio.to_thread(runtime.store.list_events,None,None,None,100)
                except Exception:events=[]
                await asyncio.wait_for(ws.send_json({"state":runtime.snapshot(),"events":events}),timeout=2)
                await asyncio.sleep(1)
        except (WebSocketDisconnect,asyncio.TimeoutError,RuntimeError):pass
    return app
