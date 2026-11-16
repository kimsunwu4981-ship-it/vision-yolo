import os
import secrets
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
                port=int(os.environ.get("SAFETY_PORT","8765")),workers=1,access_log=False,log_level="warning")
    server=uvicorn.Server(config);done=threading.Event()
    def shutdown_monitor():
        while not done.wait(.25):
            try:request=read_json(directory/"stop-request.json",{})
            except (ValueError,OSError):continue
            if request.get("instance")==instance:server.should_exit=True;break
    watcher=threading.Thread(target=shutdown_monitor,daemon=True);watcher.start()
    try:server.run()
    finally:
        done.set();lock.close()
        if read_json(directory/"service.json",{}).get("instance")==instance:(directory/"service.json").unlink(missing_ok=True)
