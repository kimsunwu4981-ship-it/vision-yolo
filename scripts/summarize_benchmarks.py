"""Summarize measured artifacts, without treating DEMO runs as camera acceptance."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from app.common import ROOT,read_json,write_json

if __name__=='__main__':
    reports=[]
    for directory in sorted((ROOT/'outputs/benchmarks').glob('run-*')):
        manifest=read_json(directory/'run_manifest.json',{})
        path=directory/'performance.csv'
        if not path.exists():continue
        rows=list(csv.DictReader(path.open(encoding='utf-8')))
        if not rows:continue
        grouped=defaultdict(list)
        for row in rows:
            if float(row['elapsed_s'])>=10:grouped[row['camera_id']].append(row)
        channels={}
        for camera,items in grouped.items():
            def values(name):return [float(r[name]) for r in items if r.get(name)]
            p95=values('p95_decode_to_encoded_ms') or values('p95_host_ms')
            channels[camera]={'inference_fps_mean':float(np.mean(values('inference_fps'))),'inference_fps_min':min(values('inference_fps')),
                'preview_fps_mean':float(np.mean(values('preview_fps'))),'decode_to_encoded_p95_max_ms':max(p95) if p95 else None,
                'observation_age_max_s':max(values('observation_age_s')),'queue_max':max(values('queue')),
                'total_drops':int(items[-1]['drops'])}
        ram=[float(r['process_ram_gb']) for r in rows if r.get('process_ram_gb')]
        final=read_json(directory/'summary.json',{})
        reports.append({'run':directory.name,'requested_duration_s':manifest.get('duration_s'),'last_sample_elapsed_s':float(rows[-1]['elapsed_s']),
            'completed':bool(final),'channels':channels,'microbatch':manifest.get('config',{}).get('runtime',{}).get('microbatch',1),
            'process_ram_start_gb':ram[0] if ram else None,'process_ram_end_gb':ram[-1] if ram else None,'process_ram_peak_gb':max(ram) if ram else None,
            'vram_peak_gb':max(float(r['vram_gb']) for r in rows if r.get('vram_gb')),
            'final_storage':final.get('final',{}).get('storage'),'final_error':final.get('final',{}).get('error'),
            'real_camera_video':False,'target_laptop':False,'browser_display_latency_ms':None,'acceptance':False})
    write_json(ROOT/'docs/benchmark-results.json',reports)
    print(json.dumps(reports,ensure_ascii=True))
