from app.common import ROOT, sha256, write_json
from app.inference import Detector


def test_empty_candidate_registry_does_not_fall_back_to_operational_models(tmp_path):
    operational=ROOT/'models/model_registry.json';digest=sha256(operational)
    candidate=tmp_path/'candidate.json';write_json(candidate,{'models':[]})
    detector=Detector({'device':'cpu'},candidate);detector.load()
    assert not detector.ready and not detector.models
    assert detector.error=='no_newly_trained_evaluated_model'
    assert sha256(operational)==digest
