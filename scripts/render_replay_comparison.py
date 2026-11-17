"""Render synchronized before/after replay predictions without altering source video."""
import argparse,json,shutil,subprocess
from fractions import Fraction
from pathlib import Path
from contextlib import nullcontext
import av,cv2,numpy as np
from app.common import read_json,write_json,sha256
from app.orientation import decode_bgr
from app.types import Frame,Detection
from app.runtime import render


def run(source,before,after,before_video=None,output_prefix=None):
    source=Path(source);before=Path(before);after=Path(after)
    digest=sha256(source)
    for directory in (before,after):
        assert read_json(directory/'inputs_manifest.json')['inputs'][0]['sha256']==digest
    rows=[[json.loads(line) for line in (directory/'tracks.jsonl').read_text().splitlines()] for directory in (before,after)]
    assert len(rows[0])==len(rows[1])
    writers=[];paths=[after/'improved-silent.mp4',after/'comparison-silent.mp4']
    frame_count=0
    with av.open(str(source)) as inp,(av.open(str(before_video)) if before_video else nullcontext()) as previous:
        input_stream=inp.streams.video[0]
        previous_frames=previous.decode(video=0) if previous is not None else None
        for index,raw in enumerate(inp.decode(input_stream)):
            pixels,transform=decode_bgr(raw);h,w=pixels.shape[:2];pts=float(raw.pts*raw.time_base)
            panels=[]
            for label,records in zip(('BEFORE','AFTER'),rows):
                record=records[index];assert record['frame_id']==index+1 and abs(record['source_pts']-pts)<1e-6
                frame=Frame(record['camera_id'],record['session_id'],index+1,w,h,pts,record['received_at_utc'],record['received_monotonic'],pixels,preprocess_transform=transform,source_mode='REPLAY')
                if label=='BEFORE' and previous_frames is not None:
                    old=next(previous_frames)
                    assert abs(float(old.pts*old.time_base)-pts)<1e-6
                    panel=old.to_ndarray(format='bgr24');assert panel.shape==pixels.shape
                else:
                    jpeg=render(frame,[Detection(**d) for d in record['tracks']],record['ppe'],[],w,True)
                    panel=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
                cv2.rectangle(panel,(0,0),(w-1,85),(20,24,32),-1)
                cv2.putText(panel,f'{label} | frame {index+1}/{len(records)} | {pts:.2f}s',(12,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA)
                cv2.putText(panel,'MODEL PREDICTIONS - same original video',(12,63),cv2.FONT_HERSHEY_SIMPLEX,.55,(180,210,225),1,cv2.LINE_AA)
                panels.append(panel)
            composites=[panels[1],np.concatenate(panels,axis=1)]
            if not writers:
                for path,canvas in zip(paths,composites):
                    container=av.open(str(path),'w');stream=container.add_stream('libx264',rate=input_stream.average_rate)
                    stream.width=canvas.shape[1];stream.height=canvas.shape[0];stream.pix_fmt='yuv420p'
                    stream.time_base=stream.codec_context.time_base=Fraction(1,90000);stream.codec_context.max_b_frames=0
                    stream.options={'crf':'20','preset':'fast','tune':'zerolatency'};writers.append((container,stream))
                first_pts=pts
            for canvas,(container,stream) in zip(composites,writers):
                encoded=av.VideoFrame.from_ndarray(canvas,format='bgr24');encoded.pts=round((pts-first_pts)*90000);encoded.time_base=Fraction(1,90000)
                for packet in stream.encode(encoded):container.mux(packet)
            if index+1 in {1,289,529}:
                ok,jpg=cv2.imencode('.jpg',composites[1]);assert ok;(after/f'comparison-{index+1:04d}.jpg').write_bytes(jpg.tobytes())
            frame_count+=1
        if previous_frames is not None:assert next(previous_frames,None) is None
    for container,stream in writers:
        for packet in stream.encode():container.mux(packet)
        container.close()
    assert frame_count==len(rows[0])
    if output_prefix is not None:
        import re
        if not re.fullmatch(r'[A-Za-z0-9_-]+',output_prefix):raise ValueError('invalid_output_prefix')
    outputs=[after/(output_prefix+'-analyzed.mp4'),after/(output_prefix+'-before-after.mp4')] if output_prefix else [after/'crane-inspection-improved.mp4',after/'crane-inspection-before-after.mp4'];qa={}
    for silent,final in zip(paths,outputs):
        subprocess.run([shutil.which('ffmpeg'),'-hide_banner','-loglevel','error','-y','-i',str(silent),'-i',str(source),
            '-map','0:v:0','-map','1:a:0?','-c','copy','-metadata:s:v:0','rotate=0','-movflags','+faststart',str(final)],check=True)
        with av.open(str(final)) as checked:
            total=0;previous=None
            for decoded in checked.decode(video=0):
                timestamp=float(decoded.pts*decoded.time_base)
                assert previous is None or timestamp>previous
                assert decoded.rotation==0;previous=timestamp;total+=1
            assert total==frame_count
            qa[final.name]={'frames':total,'width':checked.streams.video[0].width,'height':checked.streams.video[0].height,
                'duration_s':checked.duration/av.time_base,'audio_streams':len(checked.streams.audio),'pts_increasing':True,'sha256':sha256(final)}
    assert sha256(source)==digest
    write_json(after/'video-qa.json',{'source_sha256_unchanged':digest,'before_video_sha256':sha256(before_video) if before_video else None,'videos':qa});print(qa)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--before',required=True);p.add_argument('--after',required=True);p.add_argument('--before-video');p.add_argument('--output-prefix')
    a=p.parse_args();run(a.source,a.before,a.after,a.before_video,a.output_prefix)
