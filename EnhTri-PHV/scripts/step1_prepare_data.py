import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "Benchmark"
ACC_RE = re.compile(
    r"^(?:"
    r"[OPQ][0-9][A-Z0-9]{3}[0-9]"
    r"|"
    r"[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}"
    r")$"
)


def canonical_acc(x: str) -> str:
    return str(x).strip().split("-", 1)[0]


def looks_like_accession(x: str) -> bool:
    return bool(ACC_RE.match(canonical_acc(x)))


def has_cif(af_dir: Path, acc: str) -> bool:
    canon = canonical_acc(acc)
    for fn in af_dir.iterdir():
        if fn.is_file() and fn.suffix == ".cif" and canon in fn.name:
            return True
    return False


def download_alphafold(proteins_df: pd.DataFrame, af_dir: Path):
    from Bio.PDB import alphafold_db

    print("\n[2/3] Download AlphaFold mmCIF for missing canonical accessions")

    accessions = sorted(
        {
            canonical_acc(x)
            for x in proteins_df["uniprot_id"].astype(str).tolist()
            if looks_like_accession(x)
        }
    )

    ok = 0
    skipped = 0
    missing = []

    for acc in tqdm(accessions, desc="AlphaFold mmCIF"):
        if has_cif(af_dir, acc):
            skipped += 1
            continue

        try:
            preds = list(alphafold_db.get_predictions(acc))
            if not preds:
                missing.append(acc)
                continue
            cif_path = alphafold_db.download_cif_for(preds[0], directory=str(af_dir))
            if cif_path and Path(cif_path).exists():
                ok += 1
            else:
                missing.append(acc)
        except Exception:
            missing.append(acc)

    print(f"AlphaFold download complete: downloaded={ok}, skipped={skipped}, missing={len(missing)}")
    miss_txt = af_dir.parent.parent / "processed" / "alphafold_missing.txt"
    with miss_txt.open("w", encoding="utf-8") as f:
        for acc in sorted(set(missing)):
            f.write(acc + "\n")
    print(f"Saved missing list: {miss_txt}")


def read_pair_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["human_uniprot", "virus_uniprot", "label"],
        dtype={"human_uniprot": str, "virus_uniprot": str, "label": int},
    )
    df["human_uniprot"] = df["human_uniprot"].astype(str).str.strip()
    df["virus_uniprot"] = df["virus_uniprot"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    return df.drop_duplicates(subset=["human_uniprot", "virus_uniprot", "label"]).reset_index(drop=True)


def read_seq_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["protein_id", "sequence"],
        dtype=str,
    )
    df["protein_id"] = df["protein_id"].astype(str).str.strip()
    df["sequence"] = df["sequence"].fillna("").astype(str).str.strip()
    return df.drop_duplicates(subset=["protein_id"]).reset_index(drop=True)


def build_proteins_df(pair_df: pd.DataFrame, seq_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    seq_map = dict(zip(seq_df["protein_id"], seq_df["sequence"]))
    human_ids = sorted(pair_df["human_uniprot"].unique().tolist())
    virus_ids = sorted(pair_df["virus_uniprot"].unique().tolist())
    all_ids = sorted(set(human_ids) | set(virus_ids))
    missing_seq_ids = [x for x in all_ids if x not in seq_map]

    proteins = []
    for hid in human_ids:
        proteins.append(
            {
                "protein_id": f"H_{hid}",
                "species": "human",
                "uniprot_id": hid,
                "canonical_uniprot": canonical_acc(hid),
                "sequence": seq_map.get(hid, ""),
            }
        )
    for vid in virus_ids:
        proteins.append(
            {
                "protein_id": f"V_{vid}",
                "species": "virus",
                "uniprot_id": vid,
                "canonical_uniprot": canonical_acc(vid),
                "sequence": seq_map.get(vid, ""),
            }
        )
    return pd.DataFrame(proteins), missing_seq_ids


def prepare_benchmark_dataset(raw_dir: Path, out_dir: Path) -> pd.DataFrame:
    print("\n[1/3] Prepare Benchmark processed dataset files")

    seq_df = read_seq_file(raw_dir / "pro_seq.txt")
    pair_all = read_pair_file(raw_dir / "protein_pair_label.txt")
    pair_train = read_pair_file(raw_dir / "protein_pair_label_train.txt")
    pair_test = read_pair_file(raw_dir / "protein_pair_label_test.txt")

    proteins_df, missing_seq_ids = build_proteins_df(pair_all, seq_df)

    key_cols = ["human_uniprot", "virus_uniprot", "label"]
    all_keys = set(map(tuple, pair_all[key_cols].itertuples(index=False, name=None)))
    train_keys = set(map(tuple, pair_train[key_cols].itertuples(index=False, name=None)))
    test_keys = set(map(tuple, pair_test[key_cols].itertuples(index=False, name=None)))
    if train_keys & test_keys:
        raise ValueError("Benchmark train/test split contains overlapping pairs.")
    if train_keys | test_keys != all_keys:
        raise ValueError("Benchmark train/test split does not exactly cover all pairs.")

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pairs_all": out_dir / "pairs_all.csv",
        "pairs_train": out_dir / "pairs_train.csv",
        "pairs_test": out_dir / "pairs_test.csv",
        "pairs_pos": out_dir / "pairs_pos.csv",
        "pairs_neg": out_dir / "pairs_neg.csv",
        "proteins": out_dir / "proteins.csv",
        "summary": out_dir / "dataset_summary.txt",
    }

    pair_all.to_csv(paths["pairs_all"], index=False)
    pair_train.to_csv(paths["pairs_train"], index=False)
    pair_test.to_csv(paths["pairs_test"], index=False)
    pair_all[pair_all["label"] == 1].to_csv(paths["pairs_pos"], index=False)
    pair_all[pair_all["label"] == 0].to_csv(paths["pairs_neg"], index=False)
    proteins_df.to_csv(paths["proteins"], index=False)

    neg_ratio = float((pair_all["label"] == 0).sum()) / max(float((pair_all["label"] == 1).sum()), 1.0)
    with paths["summary"].open("w", encoding="utf-8") as f:
        f.write("Dataset: Benchmark\n")
        f.write("Protocol: benchmark_holdout_8_2\n")
        f.write(f"Total pairs: {len(pair_all)}\n")
        f.write(f"Positive pairs: {(pair_all['label'] == 1).sum()}\n")
        f.write(f"Negative pairs: {(pair_all['label'] == 0).sum()}\n")
        f.write(f"Negative/positive ratio: {neg_ratio:.6f}\n")
        f.write(f"Train pairs: {len(pair_train)}\n")
        f.write(f"Train positive pairs: {(pair_train['label'] == 1).sum()}\n")
        f.write(f"Train negative pairs: {(pair_train['label'] == 0).sum()}\n")
        f.write(f"Test pairs: {len(pair_test)}\n")
        f.write(f"Test positive pairs: {(pair_test['label'] == 1).sum()}\n")
        f.write(f"Test negative pairs: {(pair_test['label'] == 0).sum()}\n")
        f.write(f"Unique human proteins: {pair_all['human_uniprot'].nunique()}\n")
        f.write(f"Unique virus proteins: {pair_all['virus_uniprot'].nunique()}\n")
        f.write(f"Total unique proteins: {proteins_df['uniprot_id'].nunique()}\n")
        f.write(f"Missing sequence IDs: {len(missing_seq_ids)}\n")

    for key, path in paths.items():
        print(f"Saved {key}: {path}")

    return proteins_df


def main():
    raw_dir = PROJECT_ROOT / "data" / "raw" / DATASET_NAME
    processed_dir = PROJECT_ROOT / "data" / "processed" / DATASET_NAME
    af_dir = PROJECT_ROOT / "data" / "resources" / "alphafold_mmcif"
    af_summary_path = processed_dir / "alphafold_download_summary.json"
    af_log_path = processed_dir / "alphafold_download.log"

    print(f"Dataset: {DATASET_NAME}")
    print(f"Project root: {PROJECT_ROOT}")

    proteins_df = prepare_benchmark_dataset(raw_dir, processed_dir)
    af_dir.mkdir(parents=True, exist_ok=True)
    try:
        download_alphafold(proteins_df, af_dir)
    except ImportError as e:
        print("\n[2/3] Fallback AlphaFold download")
        print(f"Reason: {e}")
        print("Current Biopython environment does not provide Bio.PDB.alphafold_db.")
        print("Trying HTTP AlphaFold downloader instead...")
        downloader = PROJECT_ROOT / "scripts" / "download_alphafold_mmcif.py"
        result = subprocess.run(
            [sys.executable, str(downloader)],
            check=True,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        stdout_text = (result.stdout or "").strip()
        stderr_text = (result.stderr or "").strip()
        if stdout_text:
            try:
                summary = json.loads(stdout_text)
                af_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            except json.JSONDecodeError:
                af_log_path.write_text(stdout_text, encoding="utf-8")
        if stderr_text:
            af_log_path.write_text(stderr_text, encoding="utf-8")
        if af_summary_path.exists():
            print(f"Saved AlphaFold download summary: {af_summary_path}")
        if af_log_path.exists():
            print(f"Saved AlphaFold download log: {af_log_path}")
        print("AlphaFold HTTP download finished.")

    go_table_path = processed_dir / "uniprot_go.tsv"
    if go_table_path.exists():
        print(f"\n[3/3] Preserve frozen GO table: {go_table_path}")
    else:
        raise FileNotFoundError(
            f"Frozen GO table not found: {go_table_path}. "
            "Restore the released manuscript table before running feature extraction."
        )

    print(f"\nDataset-specific processed dir: {processed_dir}")
    print("\nPipeline step1 complete.")
    print("Next:")
    print("  python scripts/step2_extract_features_joblib.py")
    print("  python scripts/step3_train_lgbm.py")


if __name__ == "__main__":
    main()
