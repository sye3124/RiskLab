"""Offline pipeline regression checks; no genuine market data or downloads."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from risklab.config import Mandate
from risklab.data import read_price_csv, download_prices
from risklab.export import run_from_csv, period_return_table
from risklab.config import TICKERS


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Artificial numerical fixtures exist only in regression tests.
        rng = np.random.default_rng(77)
        n = 1600
        factors = rng.normal(0, .01, size=(n, 3))
        loadings = np.array([[.7,.1,.1],[.8,.15,.1],[.9,.2,.1],[-.1,.15,.15],
                             [.2,.25,.2],[0,0,.6],[.2,.1,.5]])
        returns = factors @ loadings.T + rng.normal(0, .002, (n, 7))
        cls.demo = pd.DataFrame(100*np.cumprod(1+returns, axis=0),
                                index=pd.bdate_range('2018-01-01', periods=n), columns=TICKERS)
        cls.demo_bytes = cls.demo.to_csv(index_label='Date').encode()

    def test_complete_export_preserves_raw_input_and_original_cleaning_report(self):
        # Leading pre-common-history observations may be trimmed; their evidence
        # must survive all the way to the exported run metadata.
        raw = self.demo.copy()
        raw.iloc[:3, 1] = np.nan
        content = raw.to_csv(index_label='Date').encode()
        expected = read_price_csv(content).attrs['data_quality']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'input_SYNTHETIC.csv'
            source.write_bytes(content)
            with patch('urllib.request.urlopen', side_effect=AssertionError('Network forbidden in offline run')):
                report = run_from_csv(source, root/'outputs', Mandate(), kind='user_supplied')
            output = report.parent
            self.assertTrue(report.is_file())
            self.assertEqual((output/'raw_input.csv').read_bytes(), content)
            metadata = json.loads((output/'run_metadata.json').read_text(encoding="utf-8"))
            self.assertEqual(metadata['original_data_quality'], expected)
            self.assertEqual(metadata['original_data_quality']['leading_rows_trimmed'], 3)
            self.assertEqual(json.loads((output/'cleaning_report.json').read_text(encoding="utf-8")), expected)
            self.assertEqual(metadata['raw_input_sha256'], hashlib.sha256(content).hexdigest())
            self.assertEqual(metadata['cleaned_prices_sha256'], hashlib.sha256((output/'cleaned_prices.csv').read_bytes()).hexdigest())
            self.assertEqual(metadata['kind'], 'user_supplied')
            self.assertTrue(metadata['source_code_sha256'])
            self.assertEqual(len(pd.read_csv(output/'metrics.csv')), 6)
            self.assertEqual(set(pd.read_csv(output/'metrics.csv')['dataset_kind']), {'user_supplied'})
            self.assertIn(2022, set(pd.read_csv(output/'calendar_year_returns.csv')['year']))
            self.assertEqual(pd.read_csv(output/'cleaned_prices.csv').shape[0], len(raw)-3)
            for filename in ('trades.csv', 'solver_diagnostics.csv', 'target_weights.csv',
                             'next_quarter_realized_risk.csv', 'wealth_index.csv', 'daily_returns.csv'):
                self.assertTrue((output/filename).is_file(), filename)
            page = report.read_text(encoding="utf-8")
            self.assertIn('USER-SUPPLIED DATA — UNVERIFIED', page)
            self.assertIn('raw_input.csv', page)
            self.assertNotIn('<script src=', page)
            self.assertNotIn('cdn.', page)
            self.assertEqual(json.loads((output/'run_status.json').read_text(encoding="utf-8"))['status'], 'complete')
            # A later failed import must not pair new raw input with old success.
            unrelated = output/'my_notes.txt'
            unrelated.write_text('preserve this user file', encoding='utf-8')
            invalid = raw.copy()
            invalid.iloc[400, 2] = np.nan
            bad_bytes = invalid.to_csv(index_label='Date').encode()
            invalid_path = root/'bad_input.csv'
            invalid_path.write_bytes(bad_bytes)
            with self.assertRaises(ValueError):
                run_from_csv(invalid_path, output, kind='user_supplied')
            self.assertFalse((output/'metrics.csv').exists())
            self.assertEqual((output/'raw_input.csv').read_bytes(), bad_bytes)
            self.assertIn('RiskLab run failed', (output/'report.html').read_text(encoding="utf-8"))
            self.assertNotIn('Net wealth index', (output/'report.html').read_text(encoding="utf-8"))
            old_reports = list((output/'previous_runs').glob('*/report.html'))
            self.assertEqual(len(old_reports), 1)
            self.assertIn('USER-SUPPLIED DATA — UNVERIFIED', old_reports[0].read_text(encoding="utf-8"))
            self.assertEqual(unrelated.read_text(encoding="utf-8"), 'preserve this user file')

    def test_missing_user_file_never_falls_back_to_demo(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'out'
            with self.assertRaises(FileNotFoundError):
                run_from_csv(Path(tmp)/'missing.csv', output, kind='user_supplied')
            self.assertFalse((output/'metrics.csv').exists())
            self.assertIn('RiskLab run failed', (output/'report.html').read_text(encoding="utf-8"))
            failure = json.loads((output/'failure_diagnostics.json').read_text(encoding="utf-8"))
            self.assertFalse(failure['synthetic_fallback_used'])
            self.assertEqual(json.loads((output/'run_status.json').read_text(encoding="utf-8"))['status'], 'failed')

    def test_invalid_internal_missing_input_is_archived_before_validation(self):
        raw = self.demo.copy()
        raw.iloc[400, 2] = np.nan
        content = raw.to_csv(index_label='Date').encode()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                read_price_csv(content, archive_dir=root)
            self.assertEqual((root/'raw_input.csv').read_bytes(), content)
            self.assertTrue((root/'failure_diagnostics.json').is_file())

    def test_provider_table_is_archived_even_when_cleaning_fails(self):
        invalid = pd.concat([self.demo, self.demo.iloc[[-1]]])
        provider = pd.concat({'Close': invalid, 'Volume': invalid*0+1}, axis=1)
        fake = types.SimpleNamespace(download=lambda **kwargs: provider)
        with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules', {'yfinance': fake}):
            root = Path(tmp)
            with self.assertRaises(ValueError):
                download_prices(archive_dir=root)
            self.assertEqual((root/'raw_provider_response.csv').read_bytes(), provider.to_csv(index_label='Date').encode())
            self.assertTrue((root/'input_prices_before_cleaning.csv').is_file())
            self.assertTrue((root/'failure_diagnostics.json').is_file())

    def test_2022_is_evaluated_without_any_2020_observation(self):
        returns = pd.DataFrame({'portfolio': [.01, -.02, .03]},
                               index=pd.to_datetime(['2021-12-31', '2022-01-03', '2022-01-04']))
        years = period_return_table(returns)
        actual = years.loc[years['year'] == 2022].iloc[0]
        self.assertEqual(actual['observations'], 2)
        self.assertAlmostEqual(actual['net_return'], .98*1.03-1)

    def test_retired_demo_is_rejected_even_if_renamed(self):
        digest = hashlib.sha256(self.demo_bytes).hexdigest()
        with patch('risklab.data._RETIRED_DEMO_SHA256', digest), tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'renamed.csv'
            path.write_bytes(self.demo_bytes)
            with self.assertRaisesRegex(ValueError, 'retired simulated'):
                read_price_csv(path)
            with self.assertRaisesRegex(ValueError, 'retired simulated'):
                run_from_csv(path, Path(tmp)/'output')
            self.assertFalse((Path(tmp)/'output'/'metrics.csv').exists())

    def test_duplicate_csv_headers_are_rejected_before_pandas_mangles_them(self):
        raw = self.demo.copy()
        raw['extra'] = raw['SPY']
        raw.columns = list(raw.columns[:-1]) + ['SPY']
        with self.assertRaises(ValueError) as caught:
            read_price_csv(raw.to_csv(index_label='Date').encode())
        self.assertEqual(caught.exception.report['duplicate_headers'], ['SPY'])

    def test_solver_failure_diagnostics_are_preserved(self):
        from risklab.data import _write_failure
        exc = ValueError('Test optimizer rejection')
        exc.diagnostics = {'solver_success': False, 'iterations': 12}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_failure(root, exc, {'kind': 'demo'})
            failure = json.loads((root/'failure_diagnostics.json').read_text(encoding="utf-8"))
            self.assertEqual(failure['solver_diagnostics'], exc.diagnostics)


if __name__ == '__main__':
    unittest.main()
