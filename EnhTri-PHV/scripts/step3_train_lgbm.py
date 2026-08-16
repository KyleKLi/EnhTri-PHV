"""Reproduce the paper-default EnhTri-PHV Benchmark experiment.

The protocol is fixed to the released pair-level 8:2 split, five-fold
stratified training, QICF reliability_v2 with arithmetic-mean pair quality,
64-dimensional Gaussian random projections with L2 normalization, CPU
LightGBM, and seed 42 (fold seeds 43--47).

Ablations, alternative fusion methods, alternative quality definitions, and
protein-level split experiments are deliberately not included here.
"""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.random_projection import GaussianRandomProjection
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "Benchmark"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / DATASET_NAME
FEATURE_PATH = PROJECT_ROOT / "outputs" / "features" / DATASET_NAME / "features.joblib"
TRAIN_PAIRS_CSV = PROCESSED_DIR / "pairs_train.csv"
TEST_PAIRS_CSV = PROCESSED_DIR / "pairs_test.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / DATASET_NAME

RANDOM_SEED = 42
N_SPLITS = 5
QICF_MODEL_NAME = "QICF"
QICF_QUALITY_VERSION = "reliability_v2"
QICF_QUALITY_KEY = "qicf_q"
QICF_QUALITY_FACTOR_NAMES = ("q_e", "q_s", "q_f")
QICF_PAIR_QUALITY_AGGREGATION = "mean"
QICF_PROJECTION_DIM = 64

QICF_EXPECTED_CONFIG = {
    "quality_version": QICF_QUALITY_VERSION,
    "pair_quality_aggregation": QICF_PAIR_QUALITY_AGGREGATION,
    "function_count_scale": 30.0,
    "function_ic_scale": 4.0,
    "structure_nn_center": 3.8,
    "structure_nn_scale": 1.5,
}

LGBM_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.02,
    "num_leaves": 255,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "min_data_in_leaf": 60,
    "reg_lambda": 3.0,
    "metric": "auc",
    "verbosity": -1,
    "device_type": "cpu",
}
NUM_BOOST_ROUND = 8000
EARLY_STOPPING_ROUNDS = 300

PROTOCOL_NAME = "benchmark_holdout_cv_ensemble"
SPLIT_STRATEGY = "provided_benchmark_8_2"
CV_SPLIT_STRATEGY = "provided_benchmark_train_5fold_cv"


def load_qicf_quality_matrix(feature_payload: dict, n_rows: int) -> tuple[np.ndarray, dict]:
    if QICF_QUALITY_KEY not in feature_payload:
        raise ValueError(
            f"Feature payload does not contain '{QICF_QUALITY_KEY}'. "
            "Regenerate it with scripts/step2_extract_features_joblib.py."
        )

    quality = np.asarray(feature_payload[QICF_QUALITY_KEY], dtype=np.float32)
    expected_shape = (n_rows, len(QICF_QUALITY_FACTOR_NAMES))
    if quality.shape != expected_shape:
        raise ValueError(f"Invalid {QICF_QUALITY_KEY} shape {quality.shape}; expected {expected_shape}.")
    if not np.all(np.isfinite(quality)) or np.any(quality < 0.0) or np.any(quality > 1.0):
        raise ValueError(f"{QICF_QUALITY_KEY} must contain finite values in [0, 1].")
    if not np.allclose(quality[:, 0], 1.0, rtol=0.0, atol=1e-7):
        raise ValueError("The paper-default QICF definition requires q_e=1 for every pair.")

    qicf_meta = feature_payload.get("meta", {}).get("qicf", {})
    configuration = qicf_meta.get("configuration", {})
    mismatches = {
        key: {"feature_payload": configuration.get(key), "required": expected}
        for key, expected in QICF_EXPECTED_CONFIG.items()
        if configuration.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "The feature payload does not use the frozen paper QICF definition: "
            f"{json.dumps(mismatches, ensure_ascii=False, sort_keys=True)}. "
            "Regenerate it with the released step2 script."
        )
    return quality, qicf_meta


def summarize_qicf_quality(quality: np.ndarray) -> dict[str, dict[str, float]]:
    summary = {}
    for index, name in enumerate(QICF_QUALITY_FACTOR_NAMES):
        values = quality[:, index]
        summary[name] = {
            "minimum": float(np.min(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "maximum": float(np.max(values)),
            "zero_fraction": float(np.mean(values <= 1e-8)),
        }
    return summary


def evaluate_at_05(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "AUROC": float(roc_auc_score(y_true, y_prob)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "ACC": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def train_lgbm(X_train, y_train, X_valid, y_valid, seed: int):
    params = dict(LGBM_PARAMS)
    params["seed"] = seed
    positives = float((y_train == 1).sum())
    negatives = float((y_train == 0).sum())
    params["scale_pos_weight"] = negatives / max(positives, 1.0)
    model = lgb.train(
        params=params,
        train_set=lgb.Dataset(X_train, label=y_train),
        valid_sets=[lgb.Dataset(X_valid, label=y_valid)],
        num_boost_round=NUM_BOOST_ROUND,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model, int(model.best_iteration), float(model.best_score["valid_0"]["auc"])


def l2_normalize_rows(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(denominator, eps)).astype(np.float32, copy=False)


def pad_projection(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] == QICF_PROJECTION_DIM:
        return matrix.astype(np.float32, copy=False)
    output = np.zeros((matrix.shape[0], QICF_PROJECTION_DIM), dtype=np.float32)
    width = min(matrix.shape[1], QICF_PROJECTION_DIM)
    output[:, :width] = matrix[:, :width].astype(np.float32, copy=False)
    return output


def fit_transform_projection(X_train, X_valid, X_predict, seed: int):
    n_components = min(QICF_PROJECTION_DIM, X_train.shape[1])
    projector = GaussianRandomProjection(n_components=n_components, random_state=seed)
    U_train = pad_projection(projector.fit_transform(X_train).astype(np.float32))
    U_valid = pad_projection(projector.transform(X_valid).astype(np.float32))
    U_predict = pad_projection(projector.transform(X_predict).astype(np.float32))
    return l2_normalize_rows(U_train), l2_normalize_rows(U_valid), l2_normalize_rows(U_predict)


def build_qicf_features(
    Xe_train,
    Xs_train,
    Xf_train,
    Xe_valid,
    Xs_valid,
    Xf_valid,
    Xe_predict,
    Xs_predict,
    Xf_predict,
    q_train,
    q_valid,
    q_predict,
    fold_seed: int,
):
    Ue_train, Ue_valid, Ue_predict = fit_transform_projection(
        Xe_train, Xe_valid, Xe_predict, seed=fold_seed + 101
    )
    Us_train, Us_valid, Us_predict = fit_transform_projection(
        Xs_train, Xs_valid, Xs_predict, seed=fold_seed + 202
    )
    Uf_train, Uf_valid, Uf_predict = fit_transform_projection(
        Xf_train, Xf_valid, Xf_predict, seed=fold_seed + 303
    )

    def assemble(Xe, Xs, Xf, Ue, Us, Uf, quality):
        quality = np.asarray(quality, dtype=np.float32)
        expected_shape = (len(Xe), len(QICF_QUALITY_FACTOR_NAMES))
        if quality.shape != expected_shape:
            raise ValueError(f"Invalid QICF quality shape {quality.shape}; expected {expected_shape}.")
        weighted = np.concatenate(
            [quality[:, [0]] * Ue, quality[:, [1]] * Us, quality[:, [2]] * Uf], axis=1
        )
        interactions = np.concatenate([Ue * Us, Ue * Uf, Us * Uf], axis=1)
        return np.concatenate([Xe, Xs, Xf, weighted, interactions, quality], axis=1).astype(np.float32)

    return (
        assemble(Xe_train, Xs_train, Xf_train, Ue_train, Us_train, Uf_train, q_train),
        assemble(Xe_valid, Xs_valid, Xf_valid, Ue_valid, Us_valid, Uf_valid, q_valid),
        assemble(Xe_predict, Xs_predict, Xf_predict, Ue_predict, Us_predict, Uf_predict, q_predict),
    )


def train_predict_fold(
    Xe_train,
    Xs_train,
    Xf_train,
    y_train,
    Xe_valid,
    Xs_valid,
    Xf_valid,
    y_valid,
    Xe_test,
    Xs_test,
    Xf_test,
    q_train,
    q_valid,
    q_test,
    fold_seed,
):
    X_train, X_valid, X_test = build_qicf_features(
        Xe_train,
        Xs_train,
        Xf_train,
        Xe_valid,
        Xs_valid,
        Xf_valid,
        Xe_test,
        Xs_test,
        Xf_test,
        q_train,
        q_valid,
        q_test,
        fold_seed,
    )
    model, best_iteration, best_auc = train_lgbm(X_train, y_train, X_valid, y_valid, fold_seed)
    valid_probability = model.predict(X_valid)
    test_probability = model.predict(X_test)
    metrics = evaluate_at_05(y_valid, valid_probability)
    row = {
        "lgb_best_iter": best_iteration,
        "lgb_best_auc": best_auc,
        "AUROC": metrics["AUROC"],
        "AUPRC": metrics["AUPRC"],
        "ACC@0.5": metrics["ACC"],
        "Precision@0.5": metrics["Precision"],
        "Recall@0.5": metrics["Recall"],
        "F1@0.5": metrics["F1"],
        "MCC@0.5": metrics["MCC"],
    }
    return row, valid_probability, test_probability


def ensure_pair_label_df(frame: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    output = frame.copy().reset_index(drop=True)
    output["label"] = output.get("label", pd.Series(labels)).astype(int).to_numpy()
    return output


def pair_key_df(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["human_uniprot"].astype(str).str.strip()
        + "\t"
        + frame["virus_uniprot"].astype(str).str.strip()
        + "\t"
        + frame["label"].astype(int).astype(str)
    )


def load_split_indices(pairs: pd.DataFrame, labels: np.ndarray):
    pairs = ensure_pair_label_df(pairs, labels)
    train_pairs = pd.read_csv(TRAIN_PAIRS_CSV)
    test_pairs = pd.read_csv(TEST_PAIRS_CSV)
    train_pairs["label"] = train_pairs["label"].astype(int)
    test_pairs["label"] = test_pairs["label"].astype(int)

    base_keys = pair_key_df(pairs)
    if base_keys.duplicated().any():
        raise ValueError(f"Feature pair table contains {int(base_keys.duplicated().sum())} duplicate keys.")
    index_by_key = pd.Series(np.arange(len(pairs)), index=base_keys)
    train_keys = pair_key_df(train_pairs)
    test_keys = pair_key_df(test_pairs)
    missing_train = [key for key in train_keys if key not in index_by_key.index]
    missing_test = [key for key in test_keys if key not in index_by_key.index]
    if missing_train or missing_test:
        raise ValueError(
            "Released Benchmark split pairs are missing from the feature payload: "
            f"train={len(missing_train)}, test={len(missing_test)}."
        )

    train_indices = index_by_key.loc[train_keys].to_numpy(dtype=int)
    test_indices = index_by_key.loc[test_keys].to_numpy(dtype=int)
    if set(train_indices) & set(test_indices):
        raise ValueError("Released Benchmark train and test indices overlap.")
    if len(train_indices) + len(test_indices) != len(pairs):
        raise ValueError("Released Benchmark train/test files do not cover the complete feature table.")
    return pairs, train_indices, test_indices


def summarize_metric_rows(rows: list[dict]) -> dict[str, float]:
    metric_keys = [
        "AUROC",
        "AUPRC",
        "ACC@0.5",
        "Precision@0.5",
        "Recall@0.5",
        "F1@0.5",
        "MCC@0.5",
        "lgb_best_iter",
    ]
    return {key: float(np.mean([row[key] for row in rows])) for key in metric_keys}


def run_benchmark_evaluation(Xe, Xs, Xf, quality, labels, pairs, train_indices, test_indices):
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    cv_splits = list(splitter.split(np.zeros(len(y_train)), y_train))
    fold_rows = []
    train_oof_probability = np.zeros(len(train_indices), dtype=np.float64)
    test_probabilities = []

    fold_bar = tqdm(enumerate(cv_splits, start=1), total=N_SPLITS, desc="Benchmark train folds")
    for fold, (fit_relative, valid_relative) in fold_bar:
        fit_indices = train_indices[np.asarray(fit_relative)]
        valid_indices = train_indices[np.asarray(valid_relative)]
        fold_seed = RANDOM_SEED + fold
        row, valid_probability, test_probability = train_predict_fold(
            Xe[fit_indices],
            Xs[fit_indices],
            Xf[fit_indices],
            labels[fit_indices],
            Xe[valid_indices],
            Xs[valid_indices],
            Xf[valid_indices],
            labels[valid_indices],
            Xe[test_indices],
            Xs[test_indices],
            Xf[test_indices],
            quality[fit_indices],
            quality[valid_indices],
            quality[test_indices],
            fold_seed,
        )
        train_oof_probability[np.asarray(valid_relative)] = valid_probability
        test_probabilities.append(test_probability.astype(np.float64))
        row.update(
            {
                "fold": fold,
                "fit_size": len(fit_indices),
                "val_size": len(valid_indices),
                "fit_pos": int(labels[fit_indices].sum()),
                "val_pos": int(labels[valid_indices].sum()),
            }
        )
        fold_rows.append(row)
        print(f"fold={fold} AUROC={row['AUROC']:.4f} AUPRC={row['AUPRC']:.4f}")
        fold_bar.set_postfix(auroc=f"{row['AUROC']:.4f}", auprc=f"{row['AUPRC']:.4f}")

    cv_summary = summarize_metric_rows(fold_rows)
    cv_summary.update(
        {
            "n_folds": N_SPLITS,
            "split_strategy": CV_SPLIT_STRATEGY,
            "fusion_mode": QICF_MODEL_NAME,
            "quality_version": QICF_QUALITY_VERSION,
            "pair_quality_aggregation": QICF_PAIR_QUALITY_AGGREGATION,
        }
    )
    oof_metrics = evaluate_at_05(y_train, train_oof_probability)
    mean_test_probability = np.mean(np.stack(test_probabilities, axis=0), axis=0)
    test_metrics = evaluate_at_05(y_test, mean_test_probability)
    final_row = {
        "protocol": PROTOCOL_NAME,
        "fusion_mode": QICF_MODEL_NAME,
        "quality_version": QICF_QUALITY_VERSION,
        "pair_quality_aggregation": QICF_PAIR_QUALITY_AGGREGATION,
        "split_strategy": SPLIT_STRATEGY,
        "seed": RANDOM_SEED,
        "n_folds": N_SPLITS,
        "train_size": len(train_indices),
        "train_pos": int(y_train.sum()),
        "train_neg": int(len(train_indices) - y_train.sum()),
        "test_size": len(test_indices),
        "test_pos": int(y_test.sum()),
        "test_neg": int(len(test_indices) - y_test.sum()),
        "cv_mean_best_iter": float(cv_summary["lgb_best_iter"]),
        "AUROC": test_metrics["AUROC"],
        "AUPRC": test_metrics["AUPRC"],
        "ACC@0.5": test_metrics["ACC"],
        "Precision@0.5": test_metrics["Precision"],
        "Recall@0.5": test_metrics["Recall"],
        "F1@0.5": test_metrics["F1"],
        "MCC@0.5": test_metrics["MCC"],
        "OOF_AUROC": oof_metrics["AUROC"],
        "OOF_AUPRC": oof_metrics["AUPRC"],
        "OOF_ACC@0.5": oof_metrics["ACC"],
        "OOF_Precision@0.5": oof_metrics["Precision"],
        "OOF_Recall@0.5": oof_metrics["Recall"],
        "OOF_F1@0.5": oof_metrics["F1"],
        "OOF_MCC@0.5": oof_metrics["MCC"],
    }
    return fold_rows, cv_summary, final_row, cv_splits


def save_split_table(pairs, train_indices, test_indices, cv_splits):
    split_table = pairs.copy().reset_index(drop=True)
    split_table["split"] = ""
    split_table["train_cv_fold"] = -1
    split_table.loc[train_indices, "split"] = "train"
    split_table.loc[test_indices, "split"] = "test"
    for fold, (_fit_relative, valid_relative) in enumerate(cv_splits, start=1):
        split_table.loc[train_indices[np.asarray(valid_relative)], "train_cv_fold"] = fold
    split_table.to_csv(OUTPUT_DIR / "benchmark_split.csv", index=False)


def save_cv_report(fold_rows, summary):
    pd.DataFrame(fold_rows).sort_values("fold").to_csv(OUTPUT_DIR / "cv_report.csv", index=False)
    joblib.dump({"folds": fold_rows, "summary": summary}, OUTPUT_DIR / "cv_report.joblib")
    with (OUTPUT_DIR / "cv_report.txt").open("w", encoding="utf-8") as handle:
        handle.write("=== Benchmark 5-Fold CV on Train Report (paper-default QICF, provided 8:2 split) ===\n\n")
        handle.write("[Fold metrics]\n")
        for row in fold_rows:
            handle.write(
                f"fold={row['fold']}  lgb_best_iter={row['lgb_best_iter']}  "
                f"lgb_best_auc={row['lgb_best_auc']:.6f}  AUROC={row['AUROC']:.4f}  "
                f"AUPRC={row['AUPRC']:.4f}  ACC@0.5={row['ACC@0.5']:.4f}  "
                f"Precision@0.5={row['Precision@0.5']:.4f}  Recall@0.5={row['Recall@0.5']:.4f}  "
                f"F1@0.5={row['F1@0.5']:.4f}  MCC@0.5={row['MCC@0.5']:.4f}\n"
            )
        handle.write("\n[CV mean]\n")
        for key, value in summary.items():
            handle.write(f"{key} = {value:.6f}\n" if isinstance(value, float) else f"{key} = {value}\n")
        handle.write("\n[Summary JSON]\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


def save_holdout_report(row):
    pd.DataFrame([row]).to_csv(OUTPUT_DIR / "holdout_test_report.csv", index=False)
    joblib.dump(row, OUTPUT_DIR / "holdout_test_report.joblib")
    with (OUTPUT_DIR / "holdout_test_report.txt").open("w", encoding="utf-8") as handle:
        handle.write("=== Benchmark Independent Test Report (paper-default QICF, provided 8:2 split) ===\n\n")
        for key, value in row.items():
            handle.write(f"{key} = {value:.6f}\n" if isinstance(value, float) else f"{key} = {value}\n")
        handle.write("\n[Summary JSON]\n" + json.dumps(row, ensure_ascii=False, indent=2) + "\n")


def main():
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = joblib.load(FEATURE_PATH)
    Xe = np.asarray(payload["X_evo"], dtype=np.float32)
    Xs = np.asarray(payload["X_struct"], dtype=np.float32)
    Xf = np.asarray(payload["X_func"], dtype=np.float32)
    labels = np.asarray(payload["y"], dtype=int)
    pairs = payload["pairs"].copy()
    if len({len(Xe), len(Xs), len(Xf), len(labels), len(pairs)}) != 1:
        raise ValueError("Feature matrices, labels, and pair rows are not aligned.")

    quality, qicf_meta = load_qicf_quality_matrix(payload, len(Xe))
    quality_summary = summarize_qicf_quality(quality)
    pairs, train_indices, test_indices = load_split_indices(pairs, labels)
    print(f"QICF quality version: {QICF_QUALITY_VERSION}")
    print(f"Pair-quality aggregation: {QICF_PAIR_QUALITY_AGGREGATION}")
    print(
        f"Benchmark 8:2 split: train={len(train_indices)} (pos={int(labels[train_indices].sum())}), "
        f"test={len(test_indices)} (pos={int(labels[test_indices].sum())})"
    )

    fold_rows, cv_summary, final_row, cv_splits = run_benchmark_evaluation(
        Xe, Xs, Xf, quality, labels, pairs, train_indices, test_indices
    )
    save_split_table(pairs, train_indices, test_indices, cv_splits)
    split_meta = {
        "dataset": DATASET_NAME,
        "protocol": PROTOCOL_NAME,
        "split_strategy": SPLIT_STRATEGY,
        "seed": RANDOM_SEED,
        "n_folds": N_SPLITS,
        "train_size": len(train_indices),
        "train_pos": int(labels[train_indices].sum()),
        "train_neg": int(len(train_indices) - labels[train_indices].sum()),
        "test_size": len(test_indices),
        "test_pos": int(labels[test_indices].sum()),
        "test_neg": int(len(test_indices) - labels[test_indices].sum()),
        "qicf_quality": qicf_meta.get("configuration", {}),
        "qicf_quality_source": "step2_feature_payload",
        "qicf_quality_summary": quality_summary,
    }
    (OUTPUT_DIR / "benchmark_split_meta.json").write_text(
        json.dumps(split_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_cv_report(fold_rows, cv_summary)
    save_holdout_report(final_row)
    print(f"Saved Benchmark reports to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
