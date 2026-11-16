"""Industrial Safety YOLO, independently implemented."""
import os
from pathlib import Path

os.environ["YOLO_CONFIG_DIR"]=str(Path(__file__).resolve().parents[1]/".cache/ultralytics")
os.environ["YOLO_AUTOINSTALL"]="False"
