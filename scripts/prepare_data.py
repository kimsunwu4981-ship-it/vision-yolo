"""Build a conservative three-class development set with group/duplicate isolation."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import yaml
from PIL import Image,ImageDraw
from app.common import ROOT,sha256,write_json


class UnionFind:
    def __init__(self,n):self.parent=list(range(n))
    def find(self,x):
        while x!=self.parent[x]:self.parent[x]=self.parent[self.parent[x]];x=self.parent[x]
        return x
    def union(self,a,b):self.parent[self.find(b)]=self.find(a)


def prepare(limit=0):
    manifests=ROOT/"data/manifests"
    with open(manifests/"dataset_inventory.csv",encoding="utf-8-sig") as stream:rows=list(csv.DictReader(stream))
    candidates=[];excluded=Counter()
    for row in rows:
        objects=json.loads(row.get("objects") or "[]");classes={o[0] for o in objects}
        if not (row["reason"].startswith("candidate_") or row["reason"] in {"development_only_human_ontology_group_isolated","excluded_exact_duplicate","excluded_partial_or_mixed_ontology"}):excluded[row["reason"]]+=1;continue
        # Only Human-annotated source family (upper-case ontology). Lower-case PPE-only sources lack people.
        if 2 not in classes or classes-{0,1,2,3,4}:row["reason"]="excluded_partial_or_mixed_ontology";excluded[row["reason"]]+=1;continue
        row["parsed_objects"]=objects;candidates.append(row)
    uf=UnionFind(len(candidates));hash_owner={};group_owner={};stem_owner={}
    for i,row in enumerate(candidates):
        # Whole-frame visual review found Video and KakaoTalk share the same rooftop site.
        family="reviewed_rooftop_site" if row["source_group"] in {"Video","KakaoTalk"} else row["source_group"]
        for lookup,key in [(hash_owner,row["sha256"]),(group_owner,family),(stem_owner,Path(row["path"]).stem.split('.rf.')[0])]:
            if key in lookup:uf.union(i,lookup[key])
            else:lookup[key]=i
    # Near-duplicate dHash <= 4: union before splitting, so look-alikes cannot cross groups.
    hashes=np.array([int(r["dhash"],16) for r in candidates],dtype=np.uint64)
    pop=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)
    near_pairs=0
    for i in range(len(candidates)):
        distances=pop[(hashes[i+1:]^hashes[i]).view(np.uint8).reshape(-1,8)].sum(axis=1)
        for j in np.flatnonzero(distances<=4):uf.union(i,i+1+int(j));near_pairs+=1
    groups=defaultdict(list)
    for i,row in enumerate(candidates):groups[uf.find(i)].append(row)
    # Largest source family goes to train; assign remaining families toward 70/15/15 targets.
    parts={"train":[],"validation":[],"test":[]};targets={"train":.7,"validation":.15,"test":.15}
    group_records=[];seen=set()
    for index,group in enumerate(sorted(groups.values(),key=lambda x:(-len(x),x[0]["path"]))):
        dest=min(parts,key=lambda key:len(parts[key])/targets[key])
        gid=hashlib.sha256("\n".join(sorted(r["sha256"] for r in group)).encode()).hexdigest()[:20]
        used=[]
        for row in group:
            if row["sha256"] in seen:row["reason"]="excluded_exact_duplicate";excluded[row["reason"]]+=1;continue
            seen.add(row["sha256"]);row["new_group"]=gid;row["new_split"]=dest;used.append(row)
        parts[dest].extend(used);group_records.append({"group":gid,"split":dest,"count":len(used),"source_families":sorted({r["source_group"] for r in group})})
    if any(not parts[key] for key in parts):raise RuntimeError("insufficient_independent_source_groups")
    mapping={1:1,2:0,4:2};names=["person","helmet","safety_vest"]
    dataset=ROOT/"data/derived/ppe_v2";class_counts={};samples=[]
    for split,selected in parts.items():
        selected.sort(key=lambda r:r["sha256"])
        if limit:selected=selected[:limit if split=="train" else max(20,limit//5)]
        records=[];counts=Counter()
        for row in selected:
            source=Path(row["path"]);name=row["sha256"][:24]
            images=dataset/split/"images";labels=dataset/split/"labels"
            images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
            target=images/(name+source.suffix)
            if not target.exists():shutil.copyfile(source,target)
            converted=[[mapping[int(c)],x,y,w,h] for c,x,y,w,h in row["parsed_objects"] if int(c) in mapping]
            label=labels/(name+".txt");label.write_text("\n".join(str(int(c))+" "+" ".join(f"{v:.8f}" for v in (x,y,w,h)) for c,x,y,w,h in converted)+"\n",encoding="utf-8")
            counts.update(names[int(o[0])] for o in converted)
            row["adopted"]=True;row["reason"]="development_only_human_ontology_group_isolated"
            record={"source":row["source"],"source_image":row["path"],"source_label":row["label"],"image_sha256":row["sha256"],
                "source_label_sha256":row["label_sha256"],"image":str(target.relative_to(ROOT)),"label":str(label.relative_to(ROOT)),
                "derived_label_sha256":sha256(label),"class_mapping":mapping,"original_split":row["split"],"split":split,
                "group":row["new_group"],"source_family":row["source_group"],"transform_version":"ppe-v1-remap-no-image-change",
                "license":"CC-BY-4.0 per source data.yaml; original contributor rights not independently verified",
                "reason":row["reason"],"field_independence_verified":False,"occlusion":"unannotated","day_night":"unannotated",
                "luminance":float(row["luminance"]),"small_objects":int(row["small_objects"])}
            records.append(record)
        write_json(manifests/(split+".json"),{"version":1,"status":"development_grouped_not_field_accepted","records":records})
        (dataset/(split+".txt")).write_text("\n".join(str((ROOT/r["image"]).resolve()) for r in records),encoding="utf-8")
        class_counts[split]=dict(counts)
        samples.extend(records[:8])
    data={"path":str(dataset.resolve()),"train":"train.txt","val":"validation.txt","test":"test.txt","names":names}
    (dataset/"data.yaml").write_text(yaml.safe_dump(data,allow_unicode=True),encoding="utf-8")
    write_json(manifests/"class_mapping.json",{"version":1,"source":"construction_ppe_v3i","selected":{1:"helmet",2:"person",4:"safety_vest"},
        "visually_reviewed":["Helmet","Human","Safety Vest","vest","hat"],"excluded":{"hat":"ambiguous; not merged", "helmet/vest":"PPE-only source without person annotations", "no hat/no vest":"semantics and negative boxes unverified","forklift":"no complete labeled source"},
        "sample_review_limit":"spot review does not certify exhaustive annotation", "target_classes":names})
    write_json(manifests/"split_groups.json",{"strategy":"source-family + original-stem + SHA256 + dHash Hamming<=4 union; disjoint group assignment", "groups":group_records,"near_pairs":near_pairs,"source_scene_time_metadata":"not available","field_test_independent":False})
    write_json(manifests/"selection_summary.json",{"images":{s:len(json.loads((manifests/(s+'.json')).read_text(encoding='utf-8'))["records"]) for s in parts},"class_counts":class_counts,"excluded":dict(excluded)})
    fields=sorted({k for row in rows for k in row if k!="parsed_objects"})
    with open(manifests/"dataset_inventory.csv","w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
    # Whole-frame sample review contact sheet, with the actual retained annotations.
    sheet=Image.new("RGB",(1280,960),"#101820")
    for i,record in enumerate(samples[:24]):
        image=Image.open(ROOT/record["image"]).convert("RGB");draw=ImageDraw.Draw(image);w,h=image.size
        for line in (ROOT/record["label"]).read_text().splitlines():
            c,x,y,bw,bh=map(float,line.split());draw.rectangle(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h),outline={0:"cyan",1:"yellow",2:"lime"}[int(c)],width=2)
        image.thumbnail((310,145));x=(i%4)*320;y=(i//4)*160;sheet.paste(image,(x,y+15));ImageDraw.Draw(sheet).text((x+2,y),record["split"]+" "+record["source_family"][:30],fill="white")
    (ROOT/"outputs/audit").mkdir(parents=True,exist_ok=True);sheet.save(ROOT/"outputs/audit/selected_frames.jpg")
    print(json.dumps({"images":{s:len(parts[s]) for s in parts},"classes":class_counts,"groups":len(groups),"near_pairs":near_pairs}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--limit",type=int,default=0);args=p.parse_args();prepare(args.limit)
