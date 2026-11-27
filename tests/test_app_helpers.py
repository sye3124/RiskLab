"""UI boundary checks. These do not claim a rendered Streamlit/browser test."""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


class AppHelperTests(unittest.TestCase):
    def test_saved_input_provenance_requires_matching_bytes(self):
        from risklab.data import local_price_provenance
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'prices.csv'
            archive = path.parent/'prices_source_archive'
            archive.mkdir()
            (archive/'cleaned_prices.csv').write_bytes(b'original')
            (archive/'source_provenance.json').write_text(json.dumps({'source':'Yahoo Finance'}))
            self.assertEqual(local_price_provenance(path, b'original')['source'], 'Yahoo Finance')
            self.assertEqual(local_price_provenance(path, b'changed')['source'], 'User-selected local CSV')

    def test_2022_is_available_even_when_2020_is_absent(self):
        returns = pd.DataFrame({"portfolio": [0.10, -0.05]},
                               index=pd.to_datetime(["2022-01-03", "2022-01-04"]))
        self.assertTrue(app.stress_window_table(returns, 2020).empty)
        table = app.stress_window_table(returns, 2022)
        self.assertAlmostEqual(table.loc["portfolio", "Net return over available dates"], 0.045)
        self.assertEqual(table.loc["portfolio", "Observations"], 2)
        self.assertEqual(table.loc["portfolio", "First observation"], "2022-01-03")

    def test_preview_executed_target_and_drifted_holdings_stay_distinct(self):
        preview = np.array([0.20, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10])
        held = np.array([0.11, 0.11, 0.11, 0.30, 0.20, 0.08, 0.09])
        target = np.array([0.10, 0.10, 0.10, 0.30, 0.20, 0.10, 0.10])
        targets = [{"strategy": "chosen", "signal_date": "2022-03-31",
                    "execution_date": "2022-04-01", **dict(zip(app.TICKERS, target))},
                   {"strategy": "chosen", "signal_date": "2022-06-30",
                    "execution_date": "2022-07-01", **dict(zip(app.TICKERS, preview))}]
        result = SimpleNamespace(
            weights={"chosen": pd.DataFrame([held], columns=app.TICKERS,
                                             index=pd.to_datetime(["2022-04-05"]))},
            trades=pd.DataFrame([
                {"strategy": "chosen", "signal_date": "2022-03-31", "date": "2022-04-01"},
                {"strategy": "chosen", "signal_date": "2022-06-30", "date": "2022-07-01"},
            ]), target_weights=pd.DataFrame(targets))
        table, dates = app.allocation_snapshot(result, "chosen", preview)
        np.testing.assert_allclose(table.iloc[:, 0], preview)
        np.testing.assert_allclose(table.iloc[:, 1], held)
        np.testing.assert_allclose(table.iloc[:, 2], target)
        self.assertEqual(dates["last_execution"], pd.Timestamp("2022-04-01"))
        self.assertEqual(dates["last_signal"], pd.Timestamp("2022-03-31"))

    def test_solver_candidates_round_trip_as_json_in_csv(self):
        candidates = [{"success": False, "message": "iteration limit"},
                      {"success": True, "iterations": 7}]
        frame = pd.DataFrame({"strategy": ["example"], "candidates": [candidates]})
        restored = pd.read_csv(io.BytesIO(app.csv_bytes(frame, index=False)))
        self.assertEqual(json.loads(restored.loc[0, "candidates"]), candidates)

    def test_grouped_risk_chart_preserves_signed_shares(self):
        st = MagicMock()
        data = pd.DataFrame({"Capital weight": [0.2]*7, "Volatility risk share": [-0.1]+[0.183333333]*6}, index=app.TICKERS)
        app.allocation_chart(st, data)
        long, spec = st.vega_lite_chart.call_args.args
        self.assertIn("xOffset", spec["layer"][0]["encoding"])
        self.assertIsNone(spec["layer"][0]["encoding"]["y"]["stack"])
        self.assertAlmostEqual(long["Share"].min(), -0.1)

    def test_holdings_chart_stacks_actual_weights_without_normalizing(self):
        st = MagicMock()
        weights = pd.DataFrame([np.zeros(7), np.full(7, 1/7)], index=pd.to_datetime(["2020-01-02", "2020-01-03"]), columns=app.TICKERS)
        app.time_chart(st, weights, "Capital allocation", holdings=True)
        long, spec = st.vega_lite_chart.call_args.args
        self.assertEqual(spec["encoding"]["y"]["stack"], "zero")
        self.assertEqual(spec["encoding"]["y"]["scale"]["domain"], [0,1])
        np.testing.assert_allclose(long.groupby("Date")["Value"].sum(), [0,1])
        self.assertEqual(spec["encoding"]["x"]["axis"]["format"], "%Y")

    def test_provenance_strips_windows_and_unix_paths(self):
        from risklab.data import safe_provenance
        for path in (r"C:\Users\Example\data\prices_source_archive", "/home/example/data/prices_source_archive"):
            safe = safe_provenance({"archive_directory": path, "input_sha256": "abc"})
            self.assertEqual(safe["archive_directory"], "prices_source_archive")
            self.assertEqual(safe["input_sha256"], "abc")



def _streamlit_controlflow_double() -> ModuleType:
    """Record calls and provide defaults; no rendering or Streamlit emulation."""
    fake = ModuleType("streamlit")
    fake.sidebar = MagicMock()
    fake.session_state = {}
    for name in ("set_page_config", "title", "caption", "header", "subheader",
                 "file_uploader", "text_input", "form", "form_submit_button",
                 "warning", "info", "error", "write", "markdown", "dataframe",
                 "download_button", "json", "expander", "bar_chart", "line_chart",
                 "area_chart", "vega_lite_chart", "metric", "spinner"):
        setattr(fake, name, MagicMock())
    fake.radio = MagicMock(side_effect=lambda label, options, **kwargs: options[0])
    fake.slider = MagicMock(side_effect=lambda label, low, high, value, *args, **kwargs: value)
    fake.number_input = MagicMock(side_effect=lambda label, **kwargs: kwargs["value"])
    fake.selectbox = MagicMock(side_effect=lambda label, options, index=0, **kwargs: options[index])
    fake.tabs = MagicMock(side_effect=lambda labels: [MagicMock() for _ in labels])
    fake.columns = MagicMock(side_effect=lambda count: [MagicMock() for _ in range(count)])
    fake.cache_data = lambda **kwargs: (lambda function: function)
    fake.button = MagicMock(return_value=True)
    fake.stop = MagicMock(side_effect=RuntimeError("App unexpectedly stopped"))
    return fake


class AppControlFlowTests(unittest.TestCase):
    @unittest.skipUnless(app.DATA_PATH.is_file(), "Historical data not available; run python run.py first")
    def test_default_historical_data_runs_and_produces_complete_download_without_streamlit(self):
        fake = _streamlit_controlflow_double()
        with patch.dict(sys.modules, {"streamlit": fake}):
            app.main()
        fake.error.assert_not_called()
        fake.stop.assert_not_called()
        self.assertEqual(len(fake.tabs.call_args.args[0]), 4)
        self.assertTrue(any("Saved historical ETF prices" in str(call.args[0])
                            for call in fake.info.call_args_list))
        prepared = fake.session_state["risklab_export"]["content"]
        with zipfile.ZipFile(io.BytesIO(prepared)) as archive:
            names = archive.namelist()
            self.assertIn("report.html", names)
            self.assertIn("metrics.csv", names)
            self.assertIn("raw_input.csv", names)
            self.assertEqual(archive.read("raw_input.csv"), app.DATA_PATH.read_bytes())

    @unittest.skipUnless(app.DATA_PATH.is_file(), "Historical data not available")
    def test_changed_strategy_window_and_cap(self):
        fake = _streamlit_controlflow_double()
        fake.button.return_value = False
        fake.selectbox.side_effect = lambda label, options, index=0, **kwargs: 252 if label.startswith("Estimation") else options[0]
        fake.slider.side_effect = lambda label, low, high, value, *a, **kw: 0.25 if label == "Maximum ETF weight" else value
        with patch.dict(sys.modules, {"streamlit": fake}):
            app.main()
        fake.error.assert_not_called()
        self.assertTrue(any("outside client limits" in str(c.args) for c in fake.write.call_args_list))
        self.assertTrue(any("2007-04-02" in str(c.args) for c in fake.write.call_args_list))

    @unittest.skipUnless(app.DATA_PATH.is_file(), "Historical data not available")
    def test_infeasible_caps_stop_with_error(self):
        fake = _streamlit_controlflow_double()
        caps = {"Maximum ETF weight":0.20, "Maximum total equities":0.20, "Maximum gold + commodities":0.25}
        fake.slider.side_effect = lambda label, low, high, value, *a, **kw: caps.get(label, value)
        with patch.dict(sys.modules, {"streamlit": fake}):
            with self.assertRaisesRegex(RuntimeError, "App unexpectedly stopped"):
                app.main()
        fake.error.assert_called()
        self.assertIn("portfolio calculation stopped", str(fake.error.call_args.args[0]))


if __name__ == "__main__":
    unittest.main()
