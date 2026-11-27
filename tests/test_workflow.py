"""Check orchestration without a live network connection or new market data."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json
import pandas as pd
from risklab.config import TICKERS, START_DATE, END_DATE
from risklab.data import ensure_prices, local_price_provenance
import run


class HistoricalWorkflowTests(unittest.TestCase):
    def test_existing_file_never_downloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'etf_prices.csv'
            path.write_bytes(b'cached')
            with patch('risklab.data.download_prices', side_effect=AssertionError('Network forbidden')):
                self.assertEqual(ensure_prices(path), path)
            self.assertEqual(path.read_bytes(), b'cached')

    def test_failed_refresh_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'etf_prices.csv'
            path.write_bytes(b'cached')
            with patch('risklab.data.download_prices', side_effect=RuntimeError('Provider unavailable')):
                with self.assertRaisesRegex(RuntimeError, 'Provider unavailable'):
                    ensure_prices(path, refresh=True)
            self.assertEqual(path.read_bytes(), b'cached')

    def test_missing_file_downloads_once_and_preserves_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'etf_prices.csv'
            def provider(start, end, *, archive_dir):
                self.assertEqual((start, end), (START_DATE, END_DATE))
                frame = pd.DataFrame(100.0, index=pd.bdate_range('2020-01-01', periods=700), columns=TICKERS)
                frame.to_csv(archive_dir/'cleaned_prices.csv', index_label='Date')
                (archive_dir/'source_provenance.json').write_text(json.dumps({'source':'Mock provider (test only)'}))
                return frame
            with patch('risklab.data.download_prices', side_effect=provider) as mock:
                self.assertEqual(ensure_prices(path), path)
                self.assertEqual(ensure_prices(path), path)
                self.assertEqual(mock.call_count, 1)
            self.assertTrue(local_price_provenance(path, path.read_bytes())['archive_cleaned_input_match'])

    def test_default_cli_runs_two_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'etf_prices.csv'
            report = Path(tmp)/'report.html'
            with patch('risklab.data.ensure_prices', return_value=path), patch('risklab.export.run_from_csv', return_value=report) as runner:
                self.assertEqual(run.main([]), 0)
                self.assertEqual([c.kwargs['mandate'].estimation_days for c in runner.call_args_list], [504,252])
                self.assertEqual([c.args[1].name for c in runner.call_args_list], ['outputs_real_504','outputs_real_252'])
                self.assertTrue(all(c.args[0] == path for c in runner.call_args_list))

    def test_explicit_input_does_not_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'custom.csv'
            with patch('risklab.data.ensure_prices', side_effect=AssertionError('Network forbidden')), patch('risklab.export.run_from_csv', return_value=Path(tmp)/'report.html') as runner:
                self.assertEqual(run.main(['--input', str(path), '--window', '252']), 0)
                self.assertEqual(runner.call_count, 1)
                self.assertEqual(runner.call_args.args[0], path)
