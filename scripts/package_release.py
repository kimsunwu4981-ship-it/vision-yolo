"""Local source + this project's verified model bundle. Never includes datasets or secrets."""
import json
import subprocess
import zipfile
from app.common import ROOT,read_json,write_json,sha256,utc_now

if __name__=='__main__':
    out=ROOT/'outputs/delivery';out.mkdir(parents=True,exist_ok=True)
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    source=out/'industrial-safety-yolo-source.zip'
    subprocess.run(['git','archive','--format=zip','--output='+str(source),'HEAD'],cwd=ROOT,check=True)
    with zipfile.ZipFile(source) as archive:
        for name in archive.namelist():
            assert not name.startswith(('.venv/','outputs/','data/derived/','.git/'))
            assert name!='.env' and not name.endswith(('.pt','.db','.mp4','.jpg'))
    models=read_json(ROOT/'models/model_registry.json')['models'];model_archive=out/'industrial-safety-yolo-models.zip'
    provenance=[]
    with zipfile.ZipFile(model_archive,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for model in models:
            path=(ROOT/model['path']).resolve()
            assert path.is_relative_to(ROOT/'models') and sha256(path)==model['sha256']
            archive.write(path,path.relative_to(ROOT).as_posix());provenance.append({'run_id':model['run_id'],'sha256':model['sha256'],'evaluation_status':model['evaluation_status']})
        for path in [ROOT/'models/model_registry.json',ROOT/'LICENSE',ROOT/'THIRD_PARTY_NOTICES']:
            archive.write(path,path.relative_to(ROOT).as_posix())
        for path in (ROOT/'models').glob('*.provenance.json'):archive.write(path,path.relative_to(ROOT).as_posix())
    report={'created_at':utc_now(),'source_commit':commit,'source_zip_sha256':sha256(source),'model_zip_sha256':sha256(model_archive),
        'models':provenance,'datasets_included':False,'credentials_included':False,'field_acceptance':False,
        'installation':'Extract source ZIP, then model ZIP into the same folder. Run scripts/setup.ps1. Configure local sources. Run scripts/start.ps1.'}
    write_json(out/'release-manifest.json',report);print(json.dumps(report))
