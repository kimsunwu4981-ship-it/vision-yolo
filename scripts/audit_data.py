"""Read-only source audit. Never approves uncertain mappings or source groups silently."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from app.common import ROOT, sha256, write_json, utc_now

DEFAULTS = ["D:/agent/vigent/data/datasets/construction_ppe_v3i", "D:/121.물류창고 내 작업 안전 데이터", "C:/Users/shgus/OneDrive/바탕 화면/VISION_현장실측사진"]


def source_group(name):
    # Remove Roboflow export hash; retain conservative source-family prefix.
    stem = name.split(".rf.")[0]
    stem = re.sub(r"_(jpg|png|jpeg)$", "", stem, flags=re.I)
    return re.sub(r"[\d_\-]+$", "", stem) or "unidentified_numeric_source"


def audit(paths, output):
    output.mkdir(parents=True, exist_ok=True)
    rows, samples, counts, duplicates = [], defaultdict(list), Counter(), defaultdict(list)
    for root in map(Path, paths):
        if not root.exists():
            rows.append(dict(path=str(root), reason="source_missing", adopted=False))
            continue
        data_yaml = root / "data.yaml"
        names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"] if data_yaml.exists() else []
        for file in sorted(root.rglob("*")):
            if not file.is_file():
                continue
            if file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov", ".zip"} and ".zip." not in file.name:
                continue
            row = dict(path=str(file), source=str(root), bytes=file.stat().st_size, adopted=False,
                       sha256="", label_sha256="", reason="", split="", source_group="", transform_version="audit-v1")
            if row["bytes"] == 0:
                row["reason"] = "zero_byte_incomplete"
            elif ".zip." in file.name:
                row["reason"] = "download_temporary_incomplete"
            elif file.suffix.lower() == ".zip":
                row["reason"] = "archive_requires_verified_extraction_and_labels"
            elif file.suffix.lower() in {".mp4", ".avi", ".mov"}:
                row["sha256"] = sha256(file)
                row["reason"] = "video_requires_event_ground_truth"
            else:
                row["sha256"] = sha256(file)
                duplicates[row["sha256"]].append(str(file))
                try:
                    with Image.open(file) as image:
                        image.load()
                        row["width"], row["height"] = image.size
                        gray = np.asarray(image.convert("L").resize((9, 8)), dtype=np.int16)
                        row["dhash"] = f'{int("".join(map(str, (gray[:, 1:] > gray[:, :-1]).astype(int).flatten())), 2):016x}'
                        row["luminance"] = round(float(gray.mean()), 2)
                except Exception:
                    row["reason"] = "decode_failed"
                    rows.append(row)
                    continue
                label = file.parent.parent / "labels" / (file.stem + ".txt")
                row["split"] = {"valid": "validation"}.get(file.parent.parent.name, file.parent.parent.name)
                row["source_group"] = source_group(file.stem) if names else file.parent.name
                if not names or not label.exists():
                    row["reason"] = "no_ground_truth_or_prediction_overlay"
                else:
                    row["label_sha256"] = sha256(label)
                    row["label"] = str(label)
                    problems, objects = [], []
                    for line_number, line in enumerate(label.read_text().splitlines(), 1):
                        try:
                            parts = line.split()
                            if len(parts) != 5:
                                raise ValueError("field_count")
                            c = int(parts[0]); x, y, w, h = map(float, parts[1:])
                            if not 0 <= c < len(names) or not np.isfinite([x,y,w,h]).all():
                                raise ValueError("class_or_nonfinite")
                            if not (0 < w <= 1 and 0 < h <= 1 and x-w/2 >= -1e-5 and y-h/2 >= -1e-5 and x+w/2 <= 1.00001 and y+h/2 <= 1.00001):
                                raise ValueError("bbox_boundary")
                            objects.append([c,x,y,w,h]); counts[names[c]] += 1
                            if len(samples[names[c]]) < 8:
                                samples[names[c]].append((str(file), [x,y,w,h]))
                        except (ValueError, IndexError) as error:
                            problems.append(f"{line_number}:{error}")
                    row["objects"] = json.dumps(objects)
                    row["small_objects"] = sum(w*h < .0025 for _,x,y,w,h in objects)
                    row["reason"] = "invalid_label:" + ";".join(problems) if problems else "candidate_pending_semantics_and_source_group_review"
            rows.append(row)
    keys = sorted({key for row in rows for key in row})
    with open(output / "dataset_inventory.csv", "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    duplicate_groups = [v for v in duplicates.values() if len(v) > 1]
    dhashes = defaultdict(list)
    for row in rows:
        if "dhash" in row: dhashes[row["dhash"]].append(row["path"])
    summary = {"audited_at": utc_now(), "records": len(rows), "classes": dict(counts),
               "reasons": dict(Counter(row["reason"].split(":")[0] for row in rows)),
               "exact_duplicate_groups": len(duplicate_groups), "exact_duplicates": duplicate_groups,
               "same_dhash_groups": [v for v in dhashes.values() if len(v)>1],
               "group_counts": dict(Counter(row["source_group"] for row in rows)),
               "occlusion_and_day_night": "not annotated; luminance is not day/night ground truth",
               "split_status": "source scene/video provenance absent; original random splits not accepted as independent field test"}
    write_json(output / "audit_summary.json", summary)
    write_json(output / "class_mapping.json", {"status":"pending_visual_review", "allowed_classes":["person","helmet","safety_vest","forklift"], "hat":"not_mapped", "sources":paths})
    for split in ["train","validation","test"]:
        write_json(output / f"{split}.json", {"version":1,"status":"pending_review","records":[]})
    for name, entries in samples.items():
        sheet = Image.new("RGB", (800, 440), "#202633"); draw = ImageDraw.Draw(sheet)
        for i, (filename, box) in enumerate(entries):
            with Image.open(filename) as image:
                image = image.convert("RGB"); iw,ih = image.size; x,y,w,h=box
                crop = image.crop((max(0,(x-w*.8)*iw),max(0,(y-h*.8)*ih),min(iw,(x+w*.8)*iw),min(ih,(y+h*.8)*ih)))
                crop.thumbnail((194,190)); ox=(i%4)*200; oy=(i//4)*220
                sheet.paste(crop,(ox,oy+22)); draw.text((ox+2,oy+3), f"{name} / {i+1}",fill="white")
        target = ROOT / "outputs/audit" / f"samples_{name.replace(' ','_')}.jpg"
        target.parent.mkdir(parents=True,exist_ok=True); sheet.save(target)
    print(json.dumps({k:v for k,v in summary.items() if k not in {"exact_duplicates","same_dhash_groups","group_counts"}},ensure_ascii=True))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--sources",nargs="+",default=DEFAULTS); parser.add_argument("--output",type=Path,default=ROOT/"data/manifests")
    args=parser.parse_args(); audit(args.sources,args.output)
