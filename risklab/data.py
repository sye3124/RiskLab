"""Read explicit inputs and preserve provenance; never invent market data."""
from __future__ import annotations
import hashlib
import io
import json
import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
import pandas as pd
from .config import TICKERS, DATA_PATH, START_DATE, END_DATE, ESTIMATION_WINDOWS


# Reject the exact retired demonstration, even if it has been renamed.
_RETIRED_DEMO_SHA256 = '04b28aee61c7949de686f1d3a75ef0bccb445a5c7286de482be3e60f7a79782c'


def reject_known_demo(content: bytes) -> None:
    if hashlib.sha256(content).hexdigest() == _RETIRED_DEMO_SHA256:
        raise ValueError('The retired simulated dataset is not accepted. Supply adjusted historical ETF prices.')


def safe_provenance(metadata: dict) -> dict:
    """Remove machine-specific archive paths while retaining provenance and fingerprints."""
    result = dict(metadata)
    if result.get("archive_directory"):
        value = str(result["archive_directory"])
        result["archive_directory"] = PureWindowsPath(value).name if "\\" in value else Path(value).name
        result["archive_location_note"] = "Archive folder name only; machine-specific path omitted."
    return result


def local_price_provenance(path: Path, content: bytes) -> dict:
    path = Path(path)
    archive = path.parent / (path.stem + '_source_archive')
    metadata = {'kind': 'user_supplied', 'input_filename': path.name,
                'source': 'User-selected local CSV', 'independently_verified_market_data': False}
    try:
        if (archive / 'cleaned_prices.csv').read_bytes() == content:
            provider = json.loads((archive / 'source_provenance.json').read_text(encoding='utf-8'))
            if isinstance(provider, dict):
                metadata.update(provider)
                metadata.update(kind='user_supplied', input_filename=path.name,
                                archive_directory=archive.name, independently_verified_market_data=False,
                                archive_cleaned_input_match=True)
    except (OSError, ValueError):
        pass
    return safe_provenance(metadata)


def ensure_prices(path=DATA_PATH, *, refresh=False, start=START_DATE, end=END_DATE) -> Path:
    """Reuse the fixed historical snapshot, downloading only when requested or absent."""
    import tempfile
    import shutil
    path = Path(path).expanduser().resolve()
    if path.is_file() and not refresh:
        return path
    if path.exists() and not path.is_file():
        raise ValueError(f'Expected a price file: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    archive = path.parent / (path.stem + '_source_archive')
    stage = Path(tempfile.mkdtemp(prefix=path.stem + '_download_', dir=path.parent))
    try:
        prices = download_prices(start, end, archive_dir=stage)
        if len(prices) <= max(ESTIMATION_WINDOWS) + 2:
            raise ValueError('Downloaded history is too short for the configured windows.')
        candidate = stage / 'cleaned_prices.csv'
        read_price_csv(candidate)
        meta = stage / 'source_provenance.json'
        provenance = json.loads(meta.read_text(encoding='utf-8'))
        provenance['archive_directory'] = archive.name
        meta.write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
        if path.is_file() or archive.exists():
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            backup = path.parent / 'previous_data' / stamp
            backup.mkdir(parents=True)
            if path.is_file():
                shutil.copy2(path, backup / path.name)
            if archive.exists():
                archive.rename(backup / archive.name)
        tmp_csv = stage / 'ready.csv'
        shutil.copy2(candidate, tmp_csv)
        tmp_csv.replace(path)
        stage.rename(archive)
    except Exception as exc:
        raise RuntimeError(f'Data download/update failed: {exc}. Inspect {stage}. No fallback data were used.') from exc
    return path


def _content_bytes(source) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, (str, Path)):
        return Path(source).expanduser().read_bytes()
    if hasattr(source, "getvalue"):
        content = source.getvalue()
    else:
        position = source.tell() if hasattr(source, "tell") else None
        content = source.read()
        if position is not None and hasattr(source, "seek"):
            source.seek(position)
    return content.encode("utf-8") if isinstance(content, str) else bytes(content)


def _write_failure(folder: Path | None, exc: Exception, metadata: dict) -> None:
    if folder is None:
        return
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "failed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "error_type": type(exc).__name__, "message": str(exc),
        "data_quality": getattr(exc, "report", {}), "source": metadata,
        "solver_diagnostics": getattr(exc, "diagnostics", {}),
        "synthetic_fallback_used": False,
    }
    (folder / "failure_diagnostics.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )


def read_price_csv(file_content, *, archive_dir=None, source_metadata=None) -> pd.DataFrame:
    """Read Date + seven ETF columns, retaining exact bytes and the cleaning report.

    Input can be bytes, a path, or a file-like object. If archive_dir is supplied,
    bytes are saved before parsing/validation, including on a failed import.
    """
    from .core import clean_prices, PriceValidationError
    content = _content_bytes(file_content)
    metadata = safe_provenance(source_metadata or {})
    metadata.setdefault("kind", "user_supplied")
    metadata.setdefault("source", "CSV supplied by the user")
    metadata["input_sha256"] = hashlib.sha256(content).hexdigest()
    metadata["input_size_bytes"] = len(content)
    metadata["independently_verified_market_data"] = False
    folder = Path(archive_dir) if archive_dir is not None else None
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "raw_input.csv").write_bytes(content)
    try:
        reject_known_demo(content)
        if metadata.get("kind") != "user_supplied":
            raise ValueError("Only historical price CSV inputs are supported.")
        header = next((row for row in csv.reader(io.StringIO(content.decode('utf-8-sig')))
                       if row and any(cell.strip() for cell in row)), [])
        counts = Counter(cell.strip().upper() for cell in header)
        duplicates = [name for name, count in counts.items()
                      if count > 1 and name in (*TICKERS, 'DATE', 'DATETIME')]
        if duplicates:
            raise PriceValidationError('Duplicate CSV price/date headers are ambiguous.',
                                       {'duplicate_headers': duplicates})
        raw = pd.read_csv(io.BytesIO(content))
        keys = [c for c in raw if str(c).strip().lower() in ("date", "datetime")]
        if len(keys) != 1:
            raise ValueError("CSV must contain exactly one Date or Datetime column.")
        clean, report = clean_prices(raw.set_index(keys[0]))
    except Exception as exc:
        _write_failure(folder, exc, metadata)
        raise
    clean.attrs.update(data_quality=report, raw_content=content, source_metadata=metadata)
    return clean


def download_prices(start=START_DATE, end=END_DATE, *, archive_dir=None) -> pd.DataFrame:
    """Explicit optional Yahoo download; no third-party or synthetic fallback.

    yfinance returns a DataFrame, not original HTTP bytes. Its complete returned
    table is serialized before selecting/cleaning prices, and labeled as such.
    """
    from .core import clean_prices
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = Path(archive_dir) if archive_dir is not None else root / "data" / f"yahoo_{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {
        "kind": "user_supplied",
        "source": "Yahoo Finance via yfinance download",
        "download_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": start, "requested_end": end, "auto_adjust": True,
        "raw_format": "CSV serialization of full provider DataFrame, not original HTTP bytes",
        "independently_verified_market_data": False,
    }
    try:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError(
                "Yahoo downloading is optional and yfinance is not installed. "
                "Install yfinance in this Python environment, or supply your own CSV. "
                "Run python -m pip install -r requirements.txt."
            ) from exc
        raw = yf.download(
            tickers=list(TICKERS), start=start, end=end, auto_adjust=True,
            actions=False, group_by="column", progress=False, threads=True,
        )
        if raw is None or raw.empty:
            raise RuntimeError("Yahoo returned no data. No replacement dataset was substituted.")
        provider_content = raw.to_csv(index_label="Date").encode("utf-8")
        (folder / "raw_provider_response.csv").write_bytes(provider_content)
        metadata["raw_provider_sha256"] = hashlib.sha256(provider_content).hexdigest()
        if not isinstance(raw.columns, pd.MultiIndex):
            raise ValueError("Unexpected Yahoo columns; inspect raw_provider_response.csv.")
        if "Close" in raw.columns.get_level_values(0):
            closes = raw["Close"]
        elif "Close" in raw.columns.get_level_values(1):
            closes = raw.xs("Close", axis=1, level=1)
        else:
            raise ValueError("The Yahoo response contains no Close series.")
        input_content = closes.to_csv(index_label="Date").encode("utf-8")
        (folder / "input_prices_before_cleaning.csv").write_bytes(input_content)
        clean, report = clean_prices(closes)
        clean.to_csv(folder / "cleaned_prices.csv", index_label="Date")
        (folder / "cleaning_report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
        )
    except Exception as exc:
        _write_failure(folder, exc, metadata)
        raise
    metadata["archive_directory"] = folder.name
    (folder / "source_provenance.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    clean.attrs.update(
        data_quality=report, raw_content=provider_content,
        analysis_input_content=input_content, source_metadata=metadata,
    )
    return clean
