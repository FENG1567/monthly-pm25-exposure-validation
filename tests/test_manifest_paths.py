"""JSON provenance checks must not depend on filename extensions."""
import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pm25_repro.analysis import _validate_manifest_path_tree,_validate_original_flags,AnalysisConfigError

class ManifestExtensionTests(unittest.TestCase):
    def test_original_argparse_abbreviations_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner=Path(tmp)/'runner.py'
            runner.write_text("parser.add_argument('--external-features')\nparser.add_argument('--out')\n")
            _validate_original_flags(runner,['--external-features','safe.csv','--out','safe'])
            for abbreviated in ('--external-feat','--ou'):
                with self.assertRaises(AnalysisConfigError):
                    _validate_original_flags(runner,['--external-features','safe.csv','--out','safe',abbreviated,'../outside'])
    def test_object_reference_with_non_json_filename_is_inspected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'inputs.json';child=root/'candidate.dat'
            manifest.write_text(json.dumps({'candidate_manifest':{'path':'candidate.dat'}}))
            child.write_text(json.dumps({'feature_path':'/private/input.csv'}))
            with self.assertRaises(AnalysisConfigError):
                _validate_manifest_path_tree(root,manifest,'input manifest')
    def test_top_level_non_json_extension_cannot_bypass_containment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'inputs.dat'
            manifest.write_text(json.dumps({'feature_path':'/private/input.csv'}))
            with self.assertRaises(AnalysisConfigError):
                _validate_manifest_path_tree(root,manifest,'input manifest')
    def test_nested_non_json_provenance_cannot_bypass_containment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'inputs.json';provenance=root/'provenance.dat'
            manifest.write_text(json.dumps({'target_provenance_path':'provenance.dat'}))
            provenance.write_text(json.dumps({'target_source_path':'/private/input.csv'}))
            with self.assertRaises(AnalysisConfigError):
                _validate_manifest_path_tree(root,manifest,'input manifest')
