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
    lines=['# 4채널 성능 실측','',
        '2026-11-16 개발 PC: Ryzen 9 9900X, RTX5070Ti desktop 약16GB, RAM61.38GB. 대상 노트북이 아니다. PyTorch2.10.0+cu128, YOLO26s 신규 PPE 모델, FP16/640, CPU PyAV H.264 디코딩, 960px JPEG 미리보기다.',
        '', '입력은 보유 test 사진으로 만든 2560×1440 H.264 정지 사진 시퀀스 4개를 반복한 DEMO다. 첫 시험은15FPS, 이후25FPS다. 실제 원본 현장 영상·C340I/NVR·사건 정확도·지게차 모델 부하는 포함하지 않는다. 브라우저는 연결하지 않았으며 서버 인코딩까지 측정했다.',
        '', '아래 FPS는 첫10초를 제외한 1초 간격 표본에서 최근5초 관측률의 평균이다. 최소값은 같은 표본의 최저값이다. 지연은 디코딩 준비→JPEG의 **이동 p95 중 최대값**이며 전체 run의 단일 p95가 아니다. p95 버퍼에는 시작 초기의 느린 표본도 남는다. host 네트워크 수신→브라우저 표시와 물리 종단 지연은 미측정이다.',
        '', '| run UTC | 채널 / batch | 요청초 / 완료 | 평균 추론 FPS 범위 | 최소 추론 FPS | 최대 이동 p95 ms | 프로세스 RAM peak GB |',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in reports:
        ch=list(r['channels'].values())
        if not ch:continue
        fps=[c['inference_fps_mean'] for c in ch]
        delays=[c['decode_to_encoded_p95_max_ms'] for c in ch if c['decode_to_encoded_p95_max_ms'] is not None]
        lines.append(f"| {r['run'].split('T')[1].split('-')[0]} | {len(ch)} / {r['microbatch']} | {r['requested_duration_s']} / {'완료' if r['completed'] else '미완료'} | {min(fps):.2f}–{max(fps):.2f} | {min(c['inference_fps_min'] for c in ch):.2f} | {max(delays):.1f} | {r['process_ram_peak_gb']:.3f} |")
    soak=next((r for r in reversed(reports) if r['requested_duration_s']==1800 and r['completed']),None)
    if soak:
        final=read_json(ROOT/'outputs/benchmarks'/soak['run']/'summary.json')['final']
        lines+=['','## 30분 완료 run', '',f"`{soak['run']}`. 마지막 표본 {soak['last_sample_elapsed_s']:.2f}초이며 요청1800초 루프를 정상 완료했다. PPE·전면 DEMO 위험구역·이벤트·증거 저장을 활성화했다. 실제 사건 검증이 아니다.",
            '', '| 카메라 | 평균 추론 FPS | 평균 미리보기 FPS | 마지막 이동 p95 ms | 마지막 PPE 판단불가 비율 |', '|---|---:|---:|---:|---:|']
        for c in final['cameras']:
            r=soak['channels'][c['id']]
            lines.append(f"| {c['id']} | {r['inference_fps_mean']:.2f} | {r['preview_fps_mean']:.2f} | {(c['decode_ready_to_encoded_ms'] or {}).get('p95',0):.1f} | {c['ppe_unknown_ratio']:.3f} |")
        lines+=['',f"프로세스 RAM 시작 {soak['process_ram_start_gb']:.3f} / 종료 {soak['process_ram_end_gb']:.3f} / 최대 {soak['process_ram_peak_gb']:.3f} GB. 전역 GPU VRAM 최대 {soak['vram_peak_gb']:.3f} GB. 마지막 시스템 RAM {final['system'].get('ram_used_gb')} GB. GPU/시스템 값에는 다른 프로세스도 포함된다. 기간 중 별도 CPU 평가·새 환경 설치·테스트와 다른 사용자 앱이 동작했으므로 독점 부하 시험이 아니다.",
            '', '저장 최종 상태: `'+json.dumps(soak['final_storage'],ensure_ascii=False)+'`. 파이프라인 오류: `'+str(soak['final_error'])+'`. 프레임 슬롯 최대1, 처리하지 못한 프레임은 드롭했다. 종료까지 채널 기아·OOM·지속적인 프로세스 RAM 증가 증거는 관측하지 못했으며 2시간 안정성을 보장하지 않는다.']
    lines+=['','## 해석과 재시험', '',
        'batch1에서 일부 구간15FPS 목표에 미달했다. 준비된 프레임만 최대4개 묶는 microbatch와 표시 주기 개선 후 평균 처리량이 증가하여 개발 기본값을4로 정했다. 느린 카메라를 기다리는 추가 batch 대기는0이다. 초기 encoder 가변 PTS 오류가 발생한094939 run은 실패 근거로 보존했고, 수정 후 실제 MP4 디코딩 검사를 통과했다. 095736 run은 최적화를 위해 조기 중단하여30분 완료로 세지 않는다.',
        '', '30분 평균만으로 정상 목표 통과를 주장하지 않는다. 최소5초 관측률은15FPS 아래 구간이 있고 일부 이동 p95는500ms를 넘었다. 시스템 RAM20GB 예산도 이 공유 개발 PC에서는 초과했다. host→browser500ms 목표는 아예 미측정이다. 모든 위험 규칙·지게차 모델·브라우저·노트북 조건을 포함한 정상 목표는 **미통과/미확인**이다. 사건 precision/recall·오경보/카메라시간·야간·가림·ID switch/fragmentation은 정답 영상이 없어 N/A다.',
        '', '2시간 연속 운용은 미실시다. 실제 입력/장비를 받은 뒤 같은 조건으로 아래 명령을 실행하고 브라우저 표시와 시계/LED 물리 지연도 별도로 측정한다. 지게차 신규 모델을 추가하면 전체 비용을 다시 검증한다.',
        '', '```powershell', '.venv/Scripts/python.exe -m scripts.benchmark --seconds 7200 --channels 4 --microbatch 4 --cameras configs/cameras.local.yaml', '```',
        '', '원시 입력 hash·구성·하드웨어·모델 hash·소스코드 hash는 각 run의 inputs_manifest/run_manifest, 표본은 performance.csv, 완료 상태는 summary.json에 있다. benchmark-results.json은 원시 CSV의 재계산 요약이다. 소스 변경 후 동일 run 결과를 새 코드의 장시간 검증으로 소급하지 않는다.']
    (ROOT/'docs/benchmark-4ch.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(reports,ensure_ascii=True))
