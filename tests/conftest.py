import numpy as np
import pytest
from app.types import Frame,Detection

@pytest.fixture
def frame_factory():
    def make(t=0,camera="cam1",session="session1",frame_id=1,width=640,height=480,brightness=150):
        return Frame(camera,session,frame_id,width,height,t,f"2026-11-16T00:00:{int(t)%60:02d}+00:00",t,np.full((height,width,3),brightness,np.uint8))
    return make

@pytest.fixture
def person():
    return Detection("person",(100.,50.,200.,400.),.9,1,True,1,0.,"cam1:session1:person:1")
