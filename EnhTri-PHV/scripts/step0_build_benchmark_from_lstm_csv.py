import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from Bio import pairwise2
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "base" / "lstm_phv_ppi_with_sequences.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "Benchmark_reconstructed"
NEGATIVE_RATIO = 10
SIMILARITY_THRESHOLD = 0.30
TEST_FRACTION = 0.20
RANDOM_SEED = 42


def seq_identity(a: str, b: str) -> float:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return 0.0
    aln = pairwise2.align.globalxx(a, b, one_alignment_only=True)[0]
    matches = float(aln.score)
    aln_len = float(max(len(aln.seqA), 1))
    return matches / aln_len


def write_pair_txt(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, header=False)


def write_seq_txt(seq_rows: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    seq_rows[["protein_id", "sequence"]].to_csv(path, sep="\t", index=False, header=False)


def maybe_clean_sequence(seq: str) -> str:
    return str(seq).strip().replace(" ", "").replace("\n", "").upper()


def build_positive_tables(df: pd.DataFrame):
    pos_df = df[["human_pro", "virus_pro"]].copy()
    pos_df.columns = ["human_uniprot", "virus_uniprot"]
    pos_df["label"] = 1

    human_df = df[["human_pro", "human_seq"]].drop_duplicates().copy()
    human_df.columns = ["protein_id", "sequence"]
    human_df["species"] = "human"

    virus_df = df[["virus_pro", "virus_seq"]].drop_duplicates().copy()
    virus_df.columns = ["protein_id", "sequence"]
    virus_df["species"] = "virus"

    seq_df = pd.concat([human_df, virus_df], axis=0).drop_duplicates(subset=["protein_id"]).reset_index(drop=True)
    seq_df["sequence"] = seq_df["sequence"].map(maybe_clean_sequence)
    return pos_df, seq_df


def compute_virus_similarity(virus_ids, virus_seq_map, sim_threshold: float):
    print(f"Computing virus-virus identity (threshold={sim_threshold})")
    ident = {v: {} for v in virus_ids}
    similar = {v: {v} for v in virus_ids}

    for i in tqdm(range(len(virus_ids)), desc="Virus similarity"):
        v1 = virus_ids[i]
        s1 = virus_seq_map.get(v1, "")
        for j in range(i, len(virus_ids)):
            v2 = virus_ids[j]
            s2 = virus_seq_map.get(v2, "")
            if v1 == v2:
                x = 1.0
            elif s1 == s2 and s1:
                x = 1.0
            else:
                x = seq_identity(s1, s2)
            ident[v1][v2] = x
            ident[v2][v1] = x
            if x >= sim_threshold:
                similar[v1].add(v2)
                similar[v2].add(v1)
    return ident, similar


def sample_negatives(pos_df: pd.DataFrame, seq_df: pd.DataFrame, neg_ratio: int, sim_threshold: float, seed: int):
    random.seed(seed)

    human_ids = sorted(pos_df["human_uniprot"].unique().tolist())
    virus_ids = sorted(pos_df["virus_uniprot"].unique().tolist())
    pos_set = set(zip(pos_df["human_uniprot"], pos_df["virus_uniprot"]))
    target_neg = neg_ratio * len(pos_set)

    pos_by_h = defaultdict(set)
    for h, v in pos_set:
        pos_by_h[h].add(v)

    virus_seq_map = (
        seq_df[seq_df["species"] == "virus"][["protein_id", "sequence"]]
        .drop_duplicates()
        .set_index("protein_id")["sequence"]
        .to_dict()
    )

    ident, similar = compute_virus_similarity(virus_ids, virus_seq_map, sim_threshold)

    forbidden_by_h = {}
    for h in human_ids:
        forb = set()
        for vpos in pos_by_h[h]:
            forb |= similar.get(vpos, {vpos})
        forbidden_by_h[h] = forb

    def virus_score(h, v):
        ps = pos_by_h.get(h, set())
        if not ps:
            return 0.0
        return max(ident[v].get(p, 0.0) for p in ps)

    neg_set = set()
    strict_cnt = 0
    relaxed_cnt = 0

    print("Phase A: strict dissimilar negatives")
    for h in tqdm(human_ids, desc="Strict negatives"):
        positives = pos_by_h[h]
        cand = [v for v in virus_ids if (v not in positives) and (v not in forbidden_by_h[h])]
        cand.sort(key=lambda v: virus_score(h, v))
        for v in cand:
            if len(neg_set) >= target_neg:
                break
            if (h, v) in pos_set or (h, v) in neg_set:
                continue
            neg_set.add((h, v))
            strict_cnt += 1
        if len(neg_set) >= target_neg:
            break

    if len(neg_set) < target_neg:
        print("Phase B: relaxed top-up")
        tries = 0
        max_tries = target_neg * 500
        while len(neg_set) < target_neg and tries < max_tries:
            tries += 1
            h = random.choice(human_ids)
            positives = pos_by_h[h]
            cand = [v for v in virus_ids if v not in positives]
            cand.sort(key=lambda v: virus_score(h, v))
            if not cand:
                continue
            if random.random() < 0.8:
                v = random.choice(cand[: max(1, len(cand) // 2)])
            else:
                v = random.choice(cand)
            if (h, v) in pos_set or (h, v) in neg_set:
                continue
            neg_set.add((h, v))
            relaxed_cnt += 1

    if len(neg_set) < target_neg:
        raise RuntimeError(f"Failed to sample enough negatives: got {len(neg_set)}, need {target_neg}")

    neg_df = pd.DataFrame(sorted(neg_set), columns=["human_uniprot", "virus_uniprot"])
    neg_df["label"] = 0
    stats = {
        "target_negatives": int(target_neg),
        "sampled_negatives": int(len(neg_df)),
        "strict_negatives": int(strict_cnt),
        "relaxed_negatives": int(relaxed_cnt),
        "sim_threshold": float(sim_threshold),
    }
    return neg_df, stats


def make_split(all_df: pd.DataFrame, test_size: float, seed: int):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(all_df[["human_uniprot", "virus_uniprot"]], all_df["label"]))
    train_df = all_df.iloc[train_idx].reset_index(drop=True)
    test_df = all_df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


def main():
    input_csv = INPUT_CSV
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    expected_cols = ["human_pro", "virus_pro", "human_seq", "virus_seq"]
    if list(df.columns) != expected_cols:
        raise ValueError(f"Unexpected columns: {list(df.columns)}; expected {expected_cols}")

    df = df.drop_duplicates(subset=["human_pro", "virus_pro"]).reset_index(drop=True)
    pos_df, seq_df = build_positive_tables(df)
    neg_df, neg_stats = sample_negatives(
        pos_df=pos_df,
        seq_df=seq_df,
        neg_ratio=NEGATIVE_RATIO,
        sim_threshold=SIMILARITY_THRESHOLD,
        seed=RANDOM_SEED,
    )

    all_df = pd.concat([pos_df, neg_df], axis=0).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    train_df, test_df = make_split(all_df, test_size=TEST_FRACTION, seed=RANDOM_SEED)

    write_seq_txt(seq_df, out_dir / "pro_seq.txt")
    write_pair_txt(pos_df, out_dir / "protein_pair_label_pos.txt")
    write_pair_txt(neg_df, out_dir / "protein_pair_label_neg.txt")
    write_pair_txt(all_df, out_dir / "protein_pair_label.txt")
    write_pair_txt(train_df, out_dir / "protein_pair_label_train.txt")
    write_pair_txt(test_df, out_dir / "protein_pair_label_test.txt")

    summary = {
        "source_csv": str(input_csv),
        "positive_pairs": int(len(pos_df)),
        "negative_pairs": int(len(neg_df)),
        "all_pairs": int(len(all_df)),
        "train_pairs": int(len(train_df)),
        "test_pairs": int(len(test_df)),
        "train_pos": int((train_df["label"] == 1).sum()),
        "train_neg": int((train_df["label"] == 0).sum()),
        "test_pos": int((test_df["label"] == 1).sum()),
        "test_neg": int((test_df["label"] == 0).sum()),
        "unique_human_proteins": int(pos_df["human_uniprot"].nunique()),
        "unique_virus_proteins": int(pos_df["virus_uniprot"].nunique()),
        "unique_proteins_total": int(seq_df["protein_id"].nunique()),
        "neg_ratio": NEGATIVE_RATIO,
        "sim_threshold": SIMILARITY_THRESHOLD,
        "test_size": TEST_FRACTION,
        "seed": RANDOM_SEED,
        "negative_sampling": neg_stats,
    }
    print("Benchmark raw dataset built.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
