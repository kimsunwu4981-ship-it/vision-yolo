import json
import platform
import subprocess
import sys
import importlib.metadata
from app.common import ROOT,write_json,utc_now


def diagnose():
    import torch,psutil
    result={"time":utc_now(),"python":sys.version,"executable":sys.executable,"os":platform.platform(),"cpu":platform.processor(),
        "ram_gb":psutil.virtual_memory().total/2**30,"torch":torch.__version__,"torch_cuda_runtime":torch.version.cuda,
        "gpu_available":torch.cuda.is_available(),"target_laptop_user_description":"라이젠 9 AI 392 / RTX 5070 Ti Laptop 12GB / RAM 32GB (not verified on this machine)",
        "packages":{name:importlib.metadata.version(name) for name in ["ultralytics","torch","torchvision","av","opencv-python","fastapi","uvicorn"]}}
    try:result["nvidia_smi"]=subprocess.run(["nvidia-smi"],capture_output=True,text=True,timeout=10).stdout
    except Exception as e:result["nvidia_smi_error"]=type(e).__name__
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            result["cpu_windows"]=winreg.QueryValueEx(key,"ProcessorNameString")[0].strip()
    except (ImportError,OSError):pass
    if torch.cuda.is_available():
        a=torch.randn((512,512),device="cuda");b=a@a;torch.cuda.synchronize()
        result.update(gpu_name=torch.cuda.get_device_name(0),gpu_capability=torch.cuda.get_device_capability(0),
            supported_architectures=torch.cuda.get_arch_list(),actual_tensor_device=str(b.device),gpu_tensor_finite=bool(torch.isfinite(b).all()),gpu_vram_gb=torch.cuda.get_device_properties(0).total_memory/2**30)
    write_json(ROOT/"outputs/diagnostics/environment.json",result)
    print(json.dumps(result,ensure_ascii=True,indent=2));return result

if __name__=="__main__":diagnose()
