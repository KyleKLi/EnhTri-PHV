import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "Benchmark"
PROTEINS_CSV = PROJECT_ROOT / "data" / "processed" / DATASET_NAME / "proteins.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "resources" / "alphafold_mmcif"
MAX_WORKERS = 4
FUTURE_TIMEOUT_SECONDS = 45
REQUEST_TIMEOUT_SECONDS = 30
API_SLEEP_SECONDS = 0.1
ACC_RE = re.compile(r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9][A-Z0-9]|[A-NR-Z][0-9]{5})$")


def canonical_acc(x: str) -> str:
    return str(x).strip().split("-", 1)[0]


def looks_like_accession(x: str) -> bool:
    return bool(ACC_RE.match(canonical_acc(x)))


def has_cif(out_dir: Path, acc: str) -> bool:
    canon = canonical_acc(acc)
    if not out_dir.exists():
        return False
    return any(canon in p.name for p in out_dir.glob("*.cif"))


def resolve_cif_url(acc: str, request_timeout: int) -> str | None:
    canon = canonical_acc(acc)
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{canon}"
    resp = requests.get(api_url, timeout=request_timeout)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    item = data[0]
    cif_url = item.get("cifUrl") or item.get("cif_url")
    if cif_url:
        return cif_url
    version = item.get("latestVersion", 4)
    return f"https://alphafold.ebi.ac.uk/files/AF-{canon}-F1-model_v{version}.cif"


def download_one(acc: str, out_dir: Path, request_timeout: int) -> tuple[str, str]:
    canon = canonical_acc(acc)
    if has_cif(out_dir, canon):
        return canon, "skip"
    try:
        cif_url = resolve_cif_url(canon, request_timeout=request_timeout)
        if not cif_url:
            return canon, "missing"
        resp = requests.get(cif_url, timeout=request_timeout)
        if resp.status_code == 404:
            return canon, "missing"
        resp.raise_for_status()
        version_match = re.search(r"model_v(\d+)\.cif", cif_url)
        version = version_match.group(1) if version_match else "4"
        out_path = out_dir / f"AF-{canon}-F1-model_v{version}.cif"
        out_path.write_bytes(resp.content)
        return canon, "ok"
    except Exception:
        return canon, "error"


def main():
    proteins_csv = PROTEINS_CSV
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    proteins = pd.read_csv(proteins_csv, dtype=str)
    accessions = sorted(
        {
            canonical_acc(x)
            for x in proteins["uniprot_id"].astype(str).tolist()
            if looks_like_accession(x)
        }
    )
    pending_accessions = [acc for acc in accessions if not has_cif(out_dir, acc)]

    stats = {"ok": 0, "skip": 0, "missing": 0, "error": 0}
    stats["skip"] = len(accessions) - len(pending_accessions)
    missing = []
    failed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {}
        for acc in pending_accessions:
            futures[ex.submit(download_one, acc, out_dir, REQUEST_TIMEOUT_SECONDS)] = acc

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"AlphaFold HTTP {DATASET_NAME}"):
            acc = futures[fut]
            try:
                _acc, status = fut.result(timeout=FUTURE_TIMEOUT_SECONDS)
            except TimeoutError:
                status = "error"
            except Exception:
                status = "error"
            stats[status] += 1
            if status == "missing":
                missing.append(acc)
            elif status == "error":
                failed.append(acc)
            time.sleep(API_SLEEP_SECONDS)

    summary = {
        "dataset": DATASET_NAME,
        "proteins_csv": str(proteins_csv),
        "out_dir": str(out_dir),
        "n_accessions": len(accessions),
        "n_pending_accessions": len(pending_accessions),
        "stats": stats,
        "missing_count": len(missing),
        "failed_count": len(failed),
        "sample_missing": missing[:50],
        "sample_failed": failed[:50],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
