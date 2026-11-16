"""Graceful shutdown by an instance token; never kill an unrelated reused PID."""
import psutil
from app.common import ROOT,read_json,write_json

if __name__=="__main__":
    state=read_json(ROOT/".runtime/service.json",{})
    if not state:raise SystemExit("No owned service is running")
    if state.get("root")!=str(ROOT):raise SystemExit("Project ownership mismatch")
    try:process=psutil.Process(state["pid"])
    except psutil.NoSuchProcess:raise SystemExit("Owned service has already exited")
    if abs(process.create_time()-state["create_time"])>.01:raise SystemExit("PID reused; refusing shutdown")
    write_json(ROOT/".runtime/stop-request.json",{"instance":state["instance"]})
    try:process.wait(timeout=30)
    except psutil.TimeoutExpired:raise SystemExit("Graceful shutdown timed out; inspect logs. No unrelated process terminated.")
    print("Owned server shut down gracefully")
