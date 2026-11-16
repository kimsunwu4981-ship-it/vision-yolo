import os
import re
from pathlib import Path
import yaml
from dotenv import load_dotenv
from app.common import ROOT,revision,sha256
from app.geometry import validate_polygon
from pydantic import BaseModel,Field,field_validator,model_validator


class CameraConfig(BaseModel):
    id:str=Field(pattern=r"^[a-zA-Z0-9_-]{1,40}$")
    name:str=Field(min_length=1,max_length=80)
    source_type:str="rtsp"
    url_env:str=""
    path:str=""
    enabled:bool=False
    loop:bool=False
    demo:bool=False
    required_ppe:list[str]=Field(default_factory=lambda:["helmet","safety_vest"])
    ppe_event_confirm_s:float=Field(default=.5,ge=0,le=30)
    ppe_event_clear_s:float=Field(default=1.,ge=0,le=30)
    calibration:dict|None=None

    @model_validator(mode="after")
    def validate_source(self):
        if self.source_type not in {"rtsp","file"}:raise ValueError("source_type")
        if self.url_env and not re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}",self.url_env):raise ValueError("url_env_name")
        if any(x not in {"helmet","safety_vest"} for x in self.required_ppe):raise ValueError("invalid_ppe_class")
        if self.source_type=="file" and ("://" in self.path or self.path.startswith("\\\\")):raise ValueError("file_source_requires_local_path")
        if self.calibration:
            from app.geometry import projected_distance
            projected_distance((0,0),(1,1),self.calibration.get("homography"))
            if self.calibration.get("validated") and (len(self.calibration.get("floor_points",[]))<4 or self.calibration.get("unit")!="metres"):
                raise ValueError("calibration_requires_ground_points_and_metres")
            if self.calibration.get("validated") and (len(self.calibration.get("source_size",[]))!=2 or min(self.calibration["source_size"])<=0):
                raise ValueError("calibration_requires_source_resolution")
        return self


class ZoneConfig(BaseModel):
    id:str=Field(pattern=r"^[a-zA-Z0-9_-]{1,40}$")
    name:str=Field(min_length=1,max_length=80)
    points:list[list[float]]
    anchor:str="foot"
    enter_s:float=Field(default=1.,ge=0,le=60)
    exit_s:float=Field(default=.8,ge=0,le=60)
    margin:float=Field(default=.005,ge=0,le=.1)
    required_ppe:list[str]=Field(default_factory=list)
    ppe_confirm_s:float=Field(default=.5,ge=0,le=60)
    ppe_clear_s:float=Field(default=1.,ge=0,le=60)
    @field_validator("points")
    @classmethod
    def points_valid(cls,v):return validate_polygon(v)
    @field_validator("anchor")
    @classmethod
    def anchor_valid(cls,v):
        if v not in {"foot","center"}:raise ValueError("anchor")
        return v
    @field_validator("required_ppe")
    @classmethod
    def ppe_valid(cls,v):
        if any(x not in {"helmet","safety_vest"} for x in v):raise ValueError("ppe")
        return v


def load_config(camera_file=None):
    load_dotenv(ROOT/".env")
    def read(name):return yaml.safe_load((ROOT/"configs"/name).read_text(encoding="utf-8"))
    runtime=read("runtime.yaml");rules=read("rules.yaml")
    path=Path(camera_file or os.environ.get("SAFETY_CAMERAS",ROOT/"configs/cameras.local.yaml"))
    if not path.exists():path=ROOT/"configs/cameras.example.yaml"
    cameras=yaml.safe_load(path.read_text(encoding="utf-8"))["cameras"]
    cameras=[CameraConfig.model_validate(c).model_dump() for c in cameras]
    if len(cameras)>4 or len({c["id"] for c in cameras})!=len(cameras):raise ValueError("requires_up_to_four_unique_cameras")
    registry=ROOT/'models/model_registry.json';model_registry_sha256=sha256(registry) if registry.exists() else None
    return {"runtime":runtime,"rules":rules,"cameras":cameras,"model_registry_sha256":model_registry_sha256,
        "revision":revision({"runtime":runtime,"rules":rules,"cameras":cameras,"model_registry_sha256":model_registry_sha256})}
