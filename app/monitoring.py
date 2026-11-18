import os
import platform
import subprocess
import threading
import time
from collections import deque
import numpy as np
import psutil


class Metrics:
    def __init__(self):
        self.inference=deque(maxlen=600);self.preview=deque(maxlen=600);self.latency=deque(maxlen=2000)
        self.process_ms=deque(maxlen=2000);self.total=0;self.last=None;self.frame_key=None;self.ppe_unknown=0.;self.gaps=deque(maxlen=2000)
        # live_preview only: how far the drawn boxes trail the displayed frame.
        self.overlay_age=deque(maxlen=2000)
    def record(self,frame,elapsed_ms):
        now=time.monotonic()
        if self.last is not None:self.gaps.append(max(0,frame.received_monotonic-self.last))
        self.inference.append(now);self.process_ms.append(elapsed_ms)
        self.last=frame.received_monotonic;self.frame_key=frame.key;self.total+=1
    def snapshot(self):
        now=time.monotonic()
        def fps(values):
            stamps=[t for t in list(values) if now-t<5]
            return round((len(stamps)-1)/(stamps[-1]-stamps[0]),2) if len(stamps)>1 else 0.
        values=list(self.latency);ages=list(self.overlay_age)
        return {"overlay_age_ms":dict(zip(("p50","p95"),map(float,np.percentile(ages,[50,95])))) if ages else None,"inference_fps":fps(self.inference),"preview_fps":fps(self.preview),"inferred_frames":self.total,
            "last_observation_age_s":None if self.last is None else round(now-self.last,3),"frame_key":self.frame_key,
            "decode_ready_to_encoded_ms":dict(zip(("p50","p95","p99"),map(float,np.percentile(values,[50,95,99])))) if values else None,
            "browser_display_latency_ms":None,"physical_end_to_end_ms":None,"ppe_unknown_ratio":self.ppe_unknown,
            "observation_gap_p95_s":float(np.percentile(list(self.gaps),95)) if self.gaps else None,
            "processing_mean_ms":float(np.mean(list(self.process_ms))) if self.process_ms else None}


class SystemMonitor:
    def __init__(self):
        self.value={};self.stop=threading.Event();self.thread=threading.Thread(target=self.run,daemon=True)
    def run(self):
        while not self.stop.is_set():
            memory=psutil.virtual_memory()
            value={"os":platform.platform(),"python":platform.python_version(),"cpu":platform.processor(),
                "ram_total_gb":round(memory.total/2**30,2),"ram_used_gb":round(memory.used/2**30,2),
                "process_ram_gb":round(psutil.Process().memory_info().rss/2**30,3),"cpu_percent":psutil.cpu_percent(),"gpu":None}
            try:
                import pynvml as nv
                nv.nvmlInit(); h=nv.nvmlDeviceGetHandleByIndex(0);m=nv.nvmlDeviceGetMemoryInfo(h)
                value["gpu"]={"name":nv.nvmlDeviceGetName(h),"driver":nv.nvmlSystemGetDriverVersion(),
                    "vram_used_gb":round(m.used/2**30,3),"vram_total_gb":round(m.total/2**30,3),"utilization_percent":nv.nvmlDeviceGetUtilizationRates(h).gpu}
                nv.nvmlShutdown()
            except Exception: pass
            self.value=value
            self.stop.wait(2)
