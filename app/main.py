import faulthandler
import os
import secrets
import sys
import threading
import psutil
import uvicorn
from dotenv import load_dotenv
from app.common import ROOT,write_json,read_json

if __name__=="__main__":
    load_dotenv(ROOT/".env")
    directory=ROOT/".runtime";directory.mkdir(exist_ok=True)
    lock=open(directory/"server.lock","a+b");lock.seek(0);lock.write(b"1");lock.flush();lock.seek(0)
    try:
        if os.name=="nt":
            import msvcrt
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:raise SystemExit("This project's server is already running")
    instance=secrets.token_urlsafe(24);process=psutil.Process()
    os.environ["SAFETY_INSTANCE_ID"]=instance
    write_json(directory/"service.json",{"pid":os.getpid(),"create_time":process.create_time(),"root":str(ROOT),"instance":instance,"port":int(os.environ.get("SAFETY_PORT","8765"))})
    config=uvicorn.Config("app.api:create_app",factory=True,host=os.environ.get("SAFETY_HOST","127.0.0.1"),
                port=int(os.environ.get("SAFETY_PORT","8765")),workers=1,access_log=False,log_level="warning",
                # An open dashboard holds its MJPEG and WebSocket connections forever;
                # without a bound a requested shutdown never completes.
                timeout_graceful_shutdown=5)
    server=uvicorn.Server(config);done=threading.Event();stop_requested=threading.Event()
    stacks=open(directory/"shutdown-stacks.log","w",encoding="utf-8")
    def forget_service():
        if read_json(directory/"service.json",{}).get("instance")==instance:(directory/"service.json").unlink(missing_ok=True)
    def shutdown_monitor():
        while not done.wait(.25):
            try:request=read_json(directory/"stop-request.json",{})
            except (ValueError,OSError):continue
            if request.get("instance")==instance:
                stop_requested.set()
                # A requested stop must end the process and release the camera and
                # GPU locks. This watchdog runs in C without the GIL, so it still
                # fires when every Python thread is blocked: after 20 s it writes all
                # thread stacks to .runtime/shutdown-stacks.log and exits the process.
                faulthandler.dump_traceback_later(20,exit=True,file=stacks)
                server.should_exit=True;break
    watcher=threading.Thread(target=shutdown_monitor,daemon=True);watcher.start()
    try:server.run()
    finally:
        done.set();lock.close();forget_service()
        if stop_requested.is_set():
            # Shutdown has completed. Leave now rather than let teardown of native
            # decoder and GPU threads keep a stopped server alive.
            faulthandler.cancel_dump_traceback_later();sys.stdout.flush();sys.stderr.flush();os._exit(0)
