"""Real H.264 decoding tests. Generated fixtures do not measure model accuracy."""
import time
import av
import numpy as np
from app.capture import Capture
from app.storage import Store


def make_video(path):
    with av.open(str(path),'w') as out:
        stream=out.add_stream('libx264',rate=25);stream.width=160;stream.height=120;stream.pix_fmt='yuv420p'
        for i in range(25):
            frame=av.VideoFrame.from_ndarray(np.full((120,160,3),i*8,np.uint8),format='bgr24')
            for packet in stream.encode(frame):out.mux(packet)
        for packet in stream.encode():out.mux(packet)


def wait_frame(capture):
    end=time.monotonic()+4
    while time.monotonic()<end:
        frame=capture.slot.take()
        if frame is not None:return frame
        time.sleep(.02)
    raise AssertionError(capture.error)


def test_real_decode_loop_seek_and_bad_camera_isolation(tmp_path):
    path=tmp_path/'sample.mp4';make_video(path)
    good=Capture(dict(id='good',source_type='file',path=str(path),loop=True))
    bad=Capture(dict(id='bad',source_type='file',path=str(tmp_path/'absent.mp4')))
    good.start();bad.start()
    try:
        first=wait_frame(good);assert first.source_pts is not None and first.image.shape==(120,160,3)
        good.seek(.5)
        end=time.monotonic()+3
        while time.monotonic()<end:
            changed=wait_frame(good)
            if changed.session_id!=first.session_id:break
        assert changed.session_id!=first.session_id
        session=changed.session_id
        while time.monotonic()<end:
            changed=wait_frame(good)
            if changed.session_id!=session:break
        assert changed.session_id!=session
        assert good.status=='connected' and bad.status in {'offline','connecting'}
    finally:good.stop();bad.stop()


def test_variable_pts_clip_round_trip(tmp_path):
    import cv2
    store=Store(tmp_path,{"min_free_gb":0})
    image=np.full((540,960,3),100,np.uint8);jpeg=cv2.imencode('.jpg',image)[1].tobytes()
    timestamps=[0,.031,.094,.14,.25,.35,.351,.7,1.2,1.3]
    store.write_evidence('variable','clip',[(t,jpeg) for t in timestamps])
    path=store.evidence_path('variable-clip')
    with av.open(str(path)) as container:
        pts=[float(f.pts*f.time_base) for f in container.decode(video=0)]
    assert len(pts)==len(timestamps)
    assert all(b>a for a,b in zip(pts,pts[1:]))
    assert abs(pts[-1]-1.3)<.01
