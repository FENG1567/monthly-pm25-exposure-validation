"""CLI must expose failed subprocess/check status to automation."""
import contextlib,io,sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pm25_repro.__main__ import main

class ExitStatusTests(unittest.TestCase):
    def test_failed_algorithm_check_is_nonzero(self):
        with patch('pm25_repro.__main__.repository_root',return_value=Path('.')),patch('pm25_repro.analysis.self_check',return_value={'ok':False,'returncode':2}),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['self-check']),2)
    def test_failed_analytical_runner_is_nonzero(self):
        with patch('pm25_repro.__main__.repository_root',return_value=Path('.')),patch('pm25_repro.analysis.run_analysis',return_value={'returncode':7}),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['analysis','--config','x.json','--component','historical','--execute']),7)
