import gzip
import json
import re
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from Bio.PDB import MMCIFParser
from tqdm import tqdm
from transformers import T5Config, T5EncoderModel, T5Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "Benchmark"
DEFAULT_PAIRS_CSV = BENCHMARK_PROCESSED_DIR / "pairs_all.csv"
DEFAULT_PROTEINS_CSV = BENCHMARK_PROCESSED_DIR / "proteins.csv"
DEFAULT_UNIPROT_GO_TSV = BENCHMARK_PROCESSED_DIR / "uniprot_go.tsv"
DEFAULT_AF_CIF_DIR = PROJECT_ROOT / "data" / "resources" / "alphafold_mmcif"
DEFAULT_GOA_GAF = PROJECT_ROOT / "data" / "resources" / "goa" / "goa_human.gaf"
DEFAULT_GOA_GAF_GZ = PROJECT_ROOT / "data" / "resources" / "goa" / "goa_human.gaf.gz"
DEFAULT_OUT_PATH = PROJECT_ROOT / "outputs" / "features" / "Benchmark" / "features.joblib"
DEFAULT_CACHE_PATH = PROJECT_ROOT / "outputs" / "features" / "Benchmark" / "prot_t5_cache.joblib"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "prot_t5_xl_half_uniref50_enc"
DEFAULT_ASCII_MODEL_DIR = DEFAULT_MODEL_DIR

CONTACT_TH = 8.0
TOPK_DEG = 64
MAX_RES = 1024
PROTT5_MAX_LEN = 512
PROTT5_BATCH_SIZE = 2
PROTT5_SAVE_EVERY = 64
RANDOM_SEED = 42
PROTT5_DEVICE = "auto"
SEQ_PAIR_MODE = "enhanced"
STRUCT_MODE = "enhanced"
FUNC_MODE = "semantic_ic"
QICF_MODEL_NAME = "QICF"
QICF_QUALITY_VERSION = "reliability_v2"
QICF_QUALITY_KEY = "qicf_q"
QICF_QUALITY_FACTOR_NAMES = ("q_e", "q_s", "q_f")
QICF_PAIR_QUALITY_AGGREGATION = "mean"
QICF_FUNCTION_COUNT_SCALE = 30.0
QICF_FUNCTION_IC_SCALE = 4.0
QICF_STRUCTURE_NN_CENTER = 3.8
QICF_STRUCTURE_NN_SCALE = 1.5

STRUCT_LOG_N_INDEX = 68
STRUCT_FRACTION_ISOLATED_INDEX = 72
STRUCT_NEAREST_NEIGHBOR_MEAN_INDEX = 74
STRUCT_COORDINATE_COVERAGE_INDEX = 79
STRUCT_MEAN_PLDDT_INDEX = 80
STRUCT_FRACTION_PLDDT_BELOW_50_INDEX = 81
STRUCT_FRACTION_PLDDT_AT_LEAST_70_INDEX = 82

STRUCTURE_RELIABILITY_FEATURE_NAMES = (
    "coordinate_coverage",
    "mean_plddt_normalized",
    "fraction_plddt_below_50",
    "fraction_plddt_at_least_70",
)

STRUCTURE_PROTEIN_FEATURE_NAMES_V2 = (
    "contact_degree_mean",
    "contact_degree_std",
    "contact_degree_min",
    "contact_degree_max",
    *(f"contact_degree_top_{index + 1}" for index in range(64)),
    "log1p_coordinate_count",
    "radius_of_gyration",
    "maximum_ca_distance",
    "contact_density",
    "fraction_isolated",
    "fraction_high_degree",
    "nearest_neighbor_distance_mean",
    "nearest_neighbor_distance_std",
    "nearest_neighbor_distance_min",
    "nearest_neighbor_distance_max",
    "sequence_hydrophobic_ratio",
    *STRUCTURE_RELIABILITY_FEATURE_NAMES,
)

HYDROPHOBIC = set("AILMFWVY")
BAD_AA_RE = re.compile(r"[UZOB]")
_CIF_INDEX_CACHE = {}


def mean_pair_quality(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Paper definition: arithmetic mean of human and virus quality."""
    left = np.clip(np.asarray(left, dtype=np.float32), 0.0, 1.0)
    right = np.clip(np.asarray(right, dtype=np.float32), 0.0, 1.0)
    combined = 0.5 * (left + right)
    return np.clip(combined, 0.0, 1.0).astype(np.float32)


def structure_protein_quality(block: np.ndarray) -> np.ndarray:
    present = (block[:, STRUCT_LOG_N_INDEX] > 0.0).astype(np.float32)
    coverage = np.clip(block[:, STRUCT_COORDINATE_COVERAGE_INDEX], 0.0, 1.0)
    mean_plddt = np.clip(block[:, STRUCT_MEAN_PLDDT_INDEX], 0.0, 1.0)
    fraction_low = np.clip(block[:, STRUCT_FRACTION_PLDDT_BELOW_50_INDEX], 0.0, 1.0)
    fraction_high = np.clip(block[:, STRUCT_FRACTION_PLDDT_AT_LEAST_70_INDEX], 0.0, 1.0)
    connected = np.clip(1.0 - block[:, STRUCT_FRACTION_ISOLATED_INDEX], 0.0, 1.0)
    nn_scale = QICF_STRUCTURE_NN_SCALE
    nn_mean = np.asarray(block[:, STRUCT_NEAREST_NEIGHBOR_MEAN_INDEX], dtype=np.float32)
    nn_plausibility = np.exp(-0.5 * ((nn_mean - QICF_STRUCTURE_NN_CENTER) / nn_scale) ** 2)
    geometry = np.sqrt(np.clip(connected * nn_plausibility, 0.0, 1.0))
    quality = (
        np.power(coverage, 0.40)
        * np.power(mean_plddt, 0.30)
        * np.power(1.0 - fraction_low, 0.15)
        * np.power(0.5 + 0.5 * fraction_high, 0.05)
        * np.power(geometry, 0.10)
    )
    return np.clip(present * quality, 0.0, 1.0).astype(np.float32)


def qicf_structure_quality(X_struct: np.ndarray) -> np.ndarray:
    protein_dim = len(STRUCTURE_PROTEIN_FEATURE_NAMES_V2)
    expected_dim = 4 * protein_dim
    if X_struct.ndim != 2 or X_struct.shape[1] != expected_dim:
        raise ValueError(f"New QICF requires a {expected_dim}-column structure matrix; got {X_struct.shape}.")
    human = X_struct[:, :protein_dim]
    virus = X_struct[:, protein_dim : 2 * protein_dim]
    q_human = structure_protein_quality(human)
    q_virus = structure_protein_quality(virus)
    return mean_pair_quality(q_human, q_virus).reshape(-1, 1)


def function_protein_quality(X_func: np.ndarray, side: int) -> np.ndarray:
    count_all = np.maximum(X_func[:, side], 0.0)
    mass_all = np.maximum(X_func[:, 3 + side], 0.0)
    namespace_counts = np.column_stack(
        [np.maximum(X_func[:, offset + side], 0.0) for offset in (10, 20, 30)]
    )
    present = (count_all > 0.0).astype(np.float32)
    count_score = -np.expm1(-count_all / QICF_FUNCTION_COUNT_SCALE)
    mean_ic = np.divide(mass_all, count_all, out=np.zeros_like(mass_all), where=count_all > 0)
    specificity_score = -np.expm1(-mean_ic / QICF_FUNCTION_IC_SCALE)
    namespace_coverage = np.mean(namespace_counts > 0.0, axis=1).astype(np.float32)
    quality = (
        np.power(np.clip(count_score, 0.0, 1.0), 0.45)
        * np.power(0.5 + 0.5 * np.clip(specificity_score, 0.0, 1.0), 0.35)
        * np.power(0.5 + 0.5 * namespace_coverage, 0.20)
    )
    return np.clip(present * quality, 0.0, 1.0).astype(np.float32)


def qicf_function_quality(X_func: np.ndarray) -> np.ndarray:
    if X_func.ndim != 2 or X_func.shape[1] != 40:
        raise ValueError(f"New QICF requires the 40-column semantic-IC function matrix; got {X_func.shape}.")
    q_human = function_protein_quality(X_func, side=0)
    q_virus = function_protein_quality(X_func, side=1)
    return mean_pair_quality(q_human, q_virus).reshape(-1, 1)


def build_qicf_quality_matrix(X_evo: np.ndarray, X_struct: np.ndarray, X_func: np.ndarray) -> np.ndarray:
    if len(X_struct) != len(X_evo) or len(X_func) != len(X_evo):
        raise ValueError("Sequence, structure, and function feature row counts must match.")
    q_e = np.ones((len(X_evo), 1), dtype=np.float32)
    q_s = qicf_structure_quality(X_struct)
    q_f = qicf_function_quality(X_func)
    return np.concatenate([q_e, q_s, q_f], axis=1).astype(np.float32)


def summarize_qicf_quality(qicf_q: np.ndarray) -> dict[str, dict[str, float]]:
    summary = {}
    for index, name in enumerate(QICF_QUALITY_FACTOR_NAMES):
        values = qicf_q[:, index]
        summary[name] = {
            "minimum": float(np.min(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "maximum": float(np.max(values)),
            "zero_fraction": float(np.mean(values <= 1e-8)),
        }
    return summary


def qicf_configuration() -> dict:
    return {
        "model_name": QICF_MODEL_NAME,
        "quality_version": QICF_QUALITY_VERSION,
        "quality_factor_names": list(QICF_QUALITY_FACTOR_NAMES),
        "pair_quality_aggregation": QICF_PAIR_QUALITY_AGGREGATION,
        "function_count_scale": QICF_FUNCTION_COUNT_SCALE,
        "function_ic_scale": QICF_FUNCTION_IC_SCALE,
        "structure_nn_center": QICF_STRUCTURE_NN_CENTER,
        "structure_nn_scale": QICF_STRUCTURE_NN_SCALE,
    }


def canonical_acc(x: str) -> str:
    return str(x).strip().split("-", 1)[0]


def hydrophobic_ratio(seq: str) -> float:
    seq = (seq or "").strip().upper()
    if not seq:
        return 0.0
    valid = [c for c in seq if c.isalpha()]
    if not valid:
        return 0.0
    hyd = sum(1 for c in valid if c in HYDROPHOBIC)
    return float(hyd) / float(len(valid))


def safe_log_ic(count: int, total: int) -> float:
    return float(-np.log((count + 1.0) / (total + 1.0)))


def load_go_namespace_map(goa_path: Path) -> dict[str, str]:
    ns_map = {}
    opener = gzip.open if goa_path.suffix == ".gz" else open
    with opener(goa_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="Parse GO namespace map"):
            if not line or line.startswith("!"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            go_id = cols[4].strip()
            aspect = cols[8].strip()
            if not go_id.startswith("GO:"):
                continue
            if aspect not in {"P", "F", "C"}:
                continue
            ns_map[go_id] = aspect
    return ns_map


def choose_goa_path(arg_value: str | None) -> Path:
    if arg_value:
        return Path(arg_value)
    if DEFAULT_GOA_GAF.exists():
        return DEFAULT_GOA_GAF
    if DEFAULT_GOA_GAF_GZ.exists():
        return DEFAULT_GOA_GAF_GZ
    raise FileNotFoundError("GOA GAF file not found. Expected data/resources/goa/goa_human.gaf or goa_human.gaf.gz")


def parse_go_terms(go_terms: str) -> set[str]:
    if not isinstance(go_terms, str) or not go_terms:
        return set()
    return {term.strip() for term in go_terms.split(";") if term.strip()}


def split_go_terms_by_namespace(go_terms: set[str], ns_map: dict[str, str]) -> tuple[set[str], set[str], set[str]]:
    bp = set()
    mf = set()
    cc = set()
    for term in go_terms:
        ns = ns_map.get(term)
        if ns == "P":
            bp.add(term)
        elif ns == "F":
            mf.add(term)
        elif ns == "C":
            cc.add(term)
    return bp, mf, cc


def build_go_ic_maps(go_df: pd.DataFrame, ns_map: dict[str, str]) -> dict[str, dict[str, float]]:
    total_proteins = 0
    total_bp = 0
    total_mf = 0
    total_cc = 0
    count_all = {}
    count_bp = {}
    count_mf = {}
    count_cc = {}

    for terms_str in go_df["go_terms"].fillna("").astype(str):
        all_terms = parse_go_terms(terms_str)
        bp_terms, mf_terms, cc_terms = split_go_terms_by_namespace(all_terms, ns_map)
        if all_terms:
            total_proteins += 1
        if bp_terms:
            total_bp += 1
        if mf_terms:
            total_mf += 1
        if cc_terms:
            total_cc += 1
        for term in all_terms:
            count_all[term] = count_all.get(term, 0) + 1
        for term in bp_terms:
            count_bp[term] = count_bp.get(term, 0) + 1
        for term in mf_terms:
            count_mf[term] = count_mf.get(term, 0) + 1
        for term in cc_terms:
            count_cc[term] = count_cc.get(term, 0) + 1

    def build_ic(counts: dict[str, int], total: int) -> dict[str, float]:
        return {term: safe_log_ic(cnt, total) for term, cnt in counts.items()}

    return {
        "all": build_ic(count_all, total_proteins),
        "bp": build_ic(count_bp, total_bp),
        "mf": build_ic(count_mf, total_mf),
        "cc": build_ic(count_cc, total_cc),
    }


def func_protein_semantic_ic(go_terms: str, ns_map: dict[str, str]):
    all_terms = parse_go_terms(go_terms)
    bp_terms, mf_terms, cc_terms = split_go_terms_by_namespace(all_terms, ns_map)
    return {
        "all": all_terms,
        "bp": bp_terms,
        "mf": mf_terms,
        "cc": cc_terms,
    }


def ic_weighted_similarity_features(a: set[str], b: set[str], ic_map: dict[str, float]) -> np.ndarray:
    inter = a & b
    union = a | b
    mass_a = float(sum(ic_map.get(term, 0.0) for term in a))
    mass_b = float(sum(ic_map.get(term, 0.0) for term in b))
    mass_inter = float(sum(ic_map.get(term, 0.0) for term in inter))
    mass_union = float(sum(ic_map.get(term, 0.0) for term in union))
    weighted_jacc = mass_inter / mass_union if mass_union > 0 else 0.0
    weighted_overlap = mass_inter / min(mass_a, mass_b) if min(mass_a, mass_b) > 0 else 0.0
    max_shared_ic = float(max((ic_map.get(term, 0.0) for term in inter), default=0.0))
    mean_shared_ic = float(mass_inter / max(len(inter), 1)) if inter else 0.0
    return np.array(
        [
            float(len(a)),
            float(len(b)),
            float(len(inter)),
            mass_a,
            mass_b,
            mass_inter,
            weighted_jacc,
            weighted_overlap,
            max_shared_ic,
            mean_shared_ic,
        ],
        dtype=np.float32,
    )


def pair_combine_func_semantic_ic(sets_a: dict[str, set[str]], sets_b: dict[str, set[str]], ic_maps: dict[str, dict[str, float]]) -> np.ndarray:
    features = [
        ic_weighted_similarity_features(sets_a["all"], sets_b["all"], ic_maps["all"]),
        ic_weighted_similarity_features(sets_a["bp"], sets_b["bp"], ic_maps["bp"]),
        ic_weighted_similarity_features(sets_a["mf"], sets_b["mf"], ic_maps["mf"]),
        ic_weighted_similarity_features(sets_a["cc"], sets_b["cc"], ic_maps["cc"]),
    ]
    return np.concatenate(features, axis=0).astype(np.float32)


def pair_combine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.concatenate([a, b, np.abs(a - b), a * b], axis=0).astype(np.float32)


def pair_combine_seq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    diff = a - b
    abs_diff = np.abs(diff)
    prod = a * b
    sq_diff = diff * diff

    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    denom = max(a_norm * b_norm, 1e-8)
    cosine = float(np.dot(a, b) / denom)

    a_unit = a / max(a_norm, 1e-8)
    b_unit = b / max(b_norm, 1e-8)
    unit_prod = a_unit * b_unit

    extra = np.array(
        [
            cosine,
            float(np.dot(a, b) / max(len(a), 1)),
            float(np.linalg.norm(diff)),
            float(abs_diff.mean()),
            float(sq_diff.mean()),
        ],
        dtype=np.float32,
    )
    return np.concatenate([a, b, abs_diff, prod, sq_diff, unit_prod, extra], axis=0).astype(np.float32)


def find_cif_for_acc(acc: str, cif_dir: Path) -> str:
    cache_key = str(cif_dir.resolve())
    if cache_key not in _CIF_INDEX_CACHE:
        exact = {}
        files = [path for path in cif_dir.iterdir() if path.is_file() and path.suffix == ".cif"]
        for path in files:
            stem = path.stem
            if stem.startswith("AF-") and "-model" in stem:
                accession = stem[3 : stem.rfind("-model")]
                accession = re.sub(r"-F\d+$", "", accession)
                exact.setdefault(accession, path)
                exact.setdefault(canonical_acc(accession), path)
        _CIF_INDEX_CACHE[cache_key] = (exact, files)

    exact, files = _CIF_INDEX_CACHE[cache_key]
    keys = [str(acc), canonical_acc(acc)]
    seen = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        if key in exact:
            return str(exact[key])
        for fn in files:
            if key in fn.name:
                return str(fn)
    return ""


def load_ca_trace_from_cif(
    cif_path: str,
    max_res: int = MAX_RES,
) -> tuple[np.ndarray, np.ndarray, int]:
    if not cif_path or not Path(cif_path).exists():
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32), 0

    parser = MMCIFParser(QUIET=True)
    cif_file = Path(cif_path)
    with cif_file.open("rb") as raw:
        is_gzip = raw.read(2) == b"\x1f\x8b"
    if is_gzip:
        with gzip.open(cif_file, "rt", encoding="utf-8") as handle:
            structure = parser.get_structure("protein", handle)
    else:
        structure = parser.get_structure("protein", str(cif_file))
    coords = []
    plddt = []
    observed_ca_count = 0
    for model in structure:
        for chain in model:
            for res in chain:
                if "CA" in res:
                    atom = res["CA"]
                    observed_ca_count += 1
                    try:
                        plddt.append(float(atom.get_bfactor()))
                    except (TypeError, ValueError):
                        plddt.append(np.nan)
                    if len(coords) < max_res:
                        coords.append(atom.coord.astype(np.float32))
        break

    if not coords:
        coord_array = np.zeros((0, 3), dtype=np.float32)
    else:
        coord_array = np.stack(coords, axis=0).astype(np.float32, copy=False)
    return coord_array, np.asarray(plddt, dtype=np.float32), observed_ca_count


def contact_degree_features(coords: np.ndarray, threshold: float, topk: int) -> np.ndarray:
    n = coords.shape[0]
    if n == 0:
        return np.zeros((4 + topk,), dtype=np.float32)

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    contact = (dist < threshold).astype(np.float32)
    np.fill_diagonal(contact, 0.0)

    degree = contact.sum(axis=1)
    degree_sorted = np.sort(degree)[::-1]
    degree_topk = degree_sorted[:topk] if len(degree_sorted) >= topk else np.pad(degree_sorted, (0, topk - len(degree_sorted)))
    stats = np.array([degree.mean(), degree.std(), degree.min(), degree.max()], dtype=np.float32)
    return np.concatenate([stats, degree_topk.astype(np.float32)], axis=0)


def struct_global_features(coords: np.ndarray, threshold: float) -> np.ndarray:
    n = coords.shape[0]
    if n == 0:
        return np.zeros((10,), dtype=np.float32)
    if n == 1:
        return np.array(
            [
                np.log1p(1.0),
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    contact = (dist < threshold).astype(np.float32)
    np.fill_diagonal(contact, 0.0)

    degree = contact.sum(axis=1)
    centered = coords - coords.mean(axis=0, keepdims=True)
    rg = float(np.sqrt(np.mean((centered ** 2).sum(axis=1))))
    max_dist = float(dist.max())

    possible_edges = max(n * (n - 1), 1)
    contact_density = float(contact.sum() / possible_edges)
    frac_isolated = float((degree == 0).mean())
    frac_high_degree = float((degree >= (degree.mean() + degree.std())).mean())

    dist_no_diag = dist.copy()
    np.fill_diagonal(dist_no_diag, np.inf)
    nn_dist = dist_no_diag.min(axis=1)
    finite_nn = nn_dist[np.isfinite(nn_dist)]
    if finite_nn.size == 0:
        finite_nn = np.zeros((1,), dtype=np.float32)

    return np.array(
        [
            np.log1p(float(n)),
            rg,
            max_dist,
            contact_density,
            frac_isolated,
            frac_high_degree,
            float(finite_nn.mean()),
            float(finite_nn.std()),
            float(finite_nn.min()),
            float(finite_nn.max()),
        ],
        dtype=np.float32,
    )


def struct_protein(acc: str, seq: str, cif_dir: Path) -> np.ndarray:
    cif_path = find_cif_for_acc(acc, cif_dir)
    coords, plddt, observed_ca_count = load_ca_trace_from_cif(cif_path, max_res=MAX_RES)
    degree_feat = contact_degree_features(coords, threshold=CONTACT_TH, topk=TOPK_DEG)
    hydro_feat = np.array([hydrophobic_ratio(seq)], dtype=np.float32)
    global_feat = struct_global_features(coords, threshold=CONTACT_TH)

    sequence_length = len(re.sub(r"[^A-Z]", "", (seq or "").upper()))
    coordinate_coverage = min(float(observed_ca_count) / float(sequence_length), 1.0) if sequence_length else 0.0
    valid_plddt = np.clip(plddt[np.isfinite(plddt)], 0.0, 100.0)
    if valid_plddt.size:
        confidence_feat = np.array(
            [
                coordinate_coverage,
                float(valid_plddt.mean() / 100.0),
                float((valid_plddt < 50.0).mean()),
                float((valid_plddt >= 70.0).mean()),
            ],
            dtype=np.float32,
        )
    else:
        confidence_feat = np.array([coordinate_coverage, 0.0, 0.0, 0.0], dtype=np.float32)

    features = np.concatenate([degree_feat, global_feat, hydro_feat, confidence_feat], axis=0).astype(np.float32)
    if len(features) != len(STRUCTURE_PROTEIN_FEATURE_NAMES_V2):
        raise RuntimeError(f"Unexpected structure feature dimension: {len(features)}")
    return features


def normalize_protein_sequence(seq: str) -> str:
    seq = (seq or "").strip().upper()
    if not seq:
        return "X"
    seq = BAD_AA_RE.sub("X", seq)
    seq = re.sub(r"[^A-Z]", "", seq)
    return " ".join(list(seq or "X"))


def path_has_non_ascii(path: Path) -> bool:
    return any(ord(ch) > 127 for ch in str(path))


def ensure_ascii_model_dir(source_dir: Path, ascii_dir: Path) -> Path:
    required = ["config.json", "pytorch_model.bin", "spiece.model", "tokenizer_config.json", "special_tokens_map.json"]
    if not source_dir.exists():
        raise FileNotFoundError(f"ProtT5 model dir not found: {source_dir}")

    if not path_has_non_ascii(source_dir):
        return source_dir

    ascii_dir.parent.mkdir(parents=True, exist_ok=True)
    ready = ascii_dir.exists() and all((ascii_dir / name).exists() for name in required)
    if not ready:
        if ascii_dir.exists():
            shutil.rmtree(ascii_dir)
        shutil.copytree(source_dir, ascii_dir)
    return ascii_dir


def load_tokenizer_and_model(model_dir: Path):
    tokenizer = T5Tokenizer.from_pretrained(str(model_dir), local_files_only=True, legacy=True)
    config = T5Config.from_json_file(str(model_dir / "config.json"))
    model = T5EncoderModel(config)
    state = torch.load(str(model_dir / "pytorch_model.bin"), map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Unexpected ProtT5 weights mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return tokenizer, model


def resolve_runtime_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in the current PyTorch environment.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("CUDA is not available. Falling back to CPU for ProtT5 encoding.")
    return torch.device("cpu")


def prepare_model_runtime(tokenizer, model, requested_device: str, max_len: int):
    preferred_device = resolve_runtime_device(requested_device)
    candidates = [preferred_device]
    if preferred_device.type == "cuda" and requested_device == "auto":
        candidates.append(torch.device("cpu"))

    last_error = None
    for device in candidates:
        try:
            model = model.to(device)
            sample = tokenizer(
                ["A"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=min(max_len, 8),
            )
            sample = {k: v.to(device) for k, v in sample.items()}
            with torch.inference_mode():
                _ = model(
                    input_ids=sample["input_ids"],
                    attention_mask=sample["attention_mask"],
                ).last_hidden_state
            return model, device
        except Exception as exc:
            last_error = exc
            if device.type == "cuda":
                print(f"CUDA warm-up failed: {exc}")
                print("Falling back to CPU for ProtT5 encoding.")
                model = model.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to initialize ProtT5 runtime device.")


def save_cache(cache_map: dict[str, np.ndarray], cache_path: Path):
    payload = {k: v.astype(np.float32) for k, v in cache_map.items()}
    joblib.dump(payload, cache_path)


def load_cache(cache_path: Path) -> dict[str, np.ndarray]:
    if not cache_path.exists():
        return {}
    obj = joblib.load(cache_path)
    return {str(k): np.asarray(v, dtype=np.float32) for k, v in obj.items()}


def encode_missing_sequences(
    seq_items,
    tokenizer,
    model,
    cache_map: dict[str, np.ndarray],
    cache_path: Path,
    batch_size: int,
    max_len: int,
    save_every: int,
    device: torch.device,
):
    missing_items = [(pid, seq) for pid, seq in seq_items if pid not in cache_map]
    if not missing_items:
        return cache_map

    processed = 0
    for start in tqdm(range(0, len(missing_items), batch_size), desc="ProtT5 encode"):
        batch = missing_items[start : start + batch_size]
        ids = [x[0] for x in batch]
        seqs = [normalize_protein_sequence(x[1]) for x in batch]
        encoded = tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled_np = pooled.cpu().numpy().astype(np.float32)
        for pid, vec in zip(ids, pooled_np):
            cache_map[pid] = vec
        processed += len(batch)
        if processed % save_every == 0:
            save_cache(cache_map, cache_path)

    save_cache(cache_map, cache_path)
    return cache_map


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    stage_bar = tqdm(total=5, desc="Step2 pipeline")

    pairs_csv = DEFAULT_PAIRS_CSV
    proteins_csv = DEFAULT_PROTEINS_CSV
    uniprot_go_tsv = DEFAULT_UNIPROT_GO_TSV
    alphafold_dir = DEFAULT_AF_CIF_DIR
    goa_gaf_path = choose_goa_path(None)
    out_path = DEFAULT_OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache_path = DEFAULT_CACHE_PATH
    summary_path = out_path.with_name(f"{out_path.stem}_summary.json")
    source_model_dir = DEFAULT_MODEL_DIR
    ascii_model_dir = DEFAULT_ASCII_MODEL_DIR
    model_dir = ensure_ascii_model_dir(source_model_dir, ascii_model_dir)

    pairs = pd.read_csv(pairs_csv)
    pairs["human_uniprot"] = pairs["human_uniprot"].astype(str)
    pairs["virus_uniprot"] = pairs["virus_uniprot"].astype(str)
    pairs["label"] = pairs["label"].astype(int)

    prots = pd.read_csv(proteins_csv)
    prots["uniprot_id"] = prots["uniprot_id"].astype(str)
    prots["canonical_uniprot"] = prots.get("canonical_uniprot", prots["uniprot_id"].map(canonical_acc)).astype(str)
    prots["sequence"] = prots["sequence"].fillna("").astype(str)
    seq_map = dict(zip(prots["uniprot_id"], prots["sequence"]))
    canonical_seq_map = dict(zip(prots["canonical_uniprot"], prots["sequence"]))

    go_df = pd.read_csv(uniprot_go_tsv, sep="\t")
    go_df["uniprot_id"] = go_df["uniprot_id"].astype(str)
    go_df["go_terms"] = go_df["go_terms"].fillna("").astype(str)
    go_map = dict(zip(go_df["uniprot_id"], go_df["go_terms"]))
    canonical_go_map = {canonical_acc(acc): terms for acc, terms in zip(go_df["uniprot_id"], go_df["go_terms"])}
    ns_map = load_go_namespace_map(goa_gaf_path)
    ic_maps = build_go_ic_maps(go_df, ns_map)
    stage_bar.update(1)
    stage_bar.set_postfix_str("loaded inputs")

    used = sorted(set(pairs["human_uniprot"]).union(set(pairs["virus_uniprot"])))
    print("Pairs:", len(pairs), "Used proteins:", len(used))

    struct_cache_path = out_path.with_name("structure_features_v2_cache.joblib")
    struct_cache = load_cache(struct_cache_path)
    struct_cache = {
        acc: feat
        for acc, feat in struct_cache.items()
        if np.asarray(feat).shape == (len(STRUCTURE_PROTEIN_FEATURE_NAMES_V2),)
    }
    func_terms_cache = {}
    print("Precompute protein features (prot/struct/func)...")
    for acc in tqdm(used):
        seq = seq_map.get(acc, canonical_seq_map.get(canonical_acc(acc), ""))
        go_terms = go_map.get(acc, canonical_go_map.get(canonical_acc(acc), ""))
        if acc not in struct_cache:
            struct_cache[acc] = struct_protein(acc, seq, alphafold_dir)
        func_terms_cache[acc] = func_protein_semantic_ic(go_terms, ns_map)
    save_cache({acc: struct_cache[acc] for acc in used}, struct_cache_path)
    print(f"Structure feature cache: {struct_cache_path}")
    stage_bar.update(1)
    stage_bar.set_postfix_str("protein features ready")

    seq_items = [(acc, seq_map.get(acc, canonical_seq_map.get(canonical_acc(acc), ""))) for acc in used]
    missing_seq = [acc for acc, seq in seq_items if not seq]
    if missing_seq:
        raise ValueError(f"Missing sequences for {len(missing_seq)} proteins, first few: {missing_seq[:5]}")

    cache_map = load_cache(cache_path)
    missing_cache_ids = [acc for acc, _seq in seq_items if acc not in cache_map]
    if missing_cache_ids:
        tokenizer, model = load_tokenizer_and_model(model_dir)
        model, runtime_device = prepare_model_runtime(tokenizer, model, PROTT5_DEVICE, PROTT5_MAX_LEN)
        print(f"ProtT5 runtime device: {runtime_device}")
        cache_map = encode_missing_sequences(
            seq_items,
            tokenizer,
            model,
            cache_map,
            cache_path=cache_path,
            batch_size=PROTT5_BATCH_SIZE,
            max_len=PROTT5_MAX_LEN,
            save_every=PROTT5_SAVE_EVERY,
            device=runtime_device,
        )
    else:
        runtime_device = "cache_only"
        print(f"ProtT5 cache complete for all {len(seq_items)} proteins; skipping tokenizer/model loading.")
    stage_bar.update(1)
    stage_bar.set_postfix_str("ProtT5 encoded")

    d_evo = len(next(iter(cache_map.values())))
    d_struct = len(next(iter(struct_cache.values())))
    probe_acc = used[0]
    d_evo_pair = len(pair_combine_seq(cache_map[probe_acc], cache_map[probe_acc]))
    d_func_pair = len(pair_combine_func_semantic_ic(func_terms_cache[probe_acc], func_terms_cache[probe_acc], ic_maps))

    num_pairs = len(pairs)
    X_evo = np.zeros((num_pairs, d_evo_pair), dtype=np.float32)
    X_struct = np.zeros((num_pairs, 4 * d_struct), dtype=np.float32)
    X_func = np.zeros((num_pairs, d_func_pair), dtype=np.float32)
    y = pairs["label"].to_numpy(dtype=np.int32)

    ids_h = pairs["human_uniprot"].tolist()
    ids_v = pairs["virus_uniprot"].tolist()

    print("Build pair matrices...")
    for i in tqdm(range(num_pairs)):
        h = ids_h[i]
        v = ids_v[i]
        X_evo[i] = pair_combine_seq(cache_map[h], cache_map[v])
        X_struct[i] = pair_combine(struct_cache[h], struct_cache[v])
        X_func[i] = pair_combine_func_semantic_ic(func_terms_cache[h], func_terms_cache[v], ic_maps)
    qicf_q = build_qicf_quality_matrix(X_evo, X_struct, X_func)
    qicf_config = qicf_configuration()
    qicf_summary = summarize_qicf_quality(qicf_q)
    stage_bar.update(1)
    stage_bar.set_postfix_str("pair matrices built")

    meta = {
        "paths": {
            "pairs": str(pairs_csv),
            "proteins": str(proteins_csv),
            "uniprot_go": str(uniprot_go_tsv),
            "alphafold_dir": str(alphafold_dir),
            "goa_gaf": str(goa_gaf_path) if goa_gaf_path else "",
            "prot_model_source_dir": str(source_model_dir),
            "prot_model_runtime_dir": str(model_dir),
            "prot_cache": str(cache_path),
            "structure_cache": str(struct_cache_path),
        },
        "params": {
            "contact_th": CONTACT_TH,
            "topk_deg": TOPK_DEG,
            "max_res": MAX_RES,
            "prot_max_len": PROTT5_MAX_LEN,
            "prot_batch_size": PROTT5_BATCH_SIZE,
            "prot_save_every": PROTT5_SAVE_EVERY,
            "device": str(runtime_device),
            "struct_mode": STRUCT_MODE,
            "func_mode": FUNC_MODE,
        },
        "dims": {
            "evo_pair": int(X_evo.shape[1]),
            "struct_protein": int(d_struct),
            "struct_pair": int(X_struct.shape[1]),
            "func_pair": int(X_func.shape[1]),
        },
        "feature_schema": {
            "structure_protein_feature_names": list(STRUCTURE_PROTEIN_FEATURE_NAMES_V2),
            "structure_reliability_feature_names": list(STRUCTURE_RELIABILITY_FEATURE_NAMES),
            "structure_pair_layout": ["human", "virus", "absolute_difference", "product"],
            "function_pair_layout": "semantic_ic_4_namespaces_x_10_features",
        },
        "counts": {
            "pairs": int(num_pairs),
            "pos": int((y == 1).sum()),
            "neg": int((y == 0).sum()),
            "used_proteins": int(len(used)),
        },
        "evo_source": "prot_t5_embedding",
        "seq_pair_mode": SEQ_PAIR_MODE,
        "plm_sequence_encoder": {
            "model_name": "Rostlab/prot_t5_xl_half_uniref50-enc",
            "embedding_dim": int(d_evo),
            "pooling": "mean_last_hidden_state",
        },
        "func_source": "go_semantic_ic",
        "qicf": {
            "model_name": QICF_MODEL_NAME,
            "configuration": qicf_config,
            "quality_summary": qicf_summary,
            "payload_key": QICF_QUALITY_KEY,
        },
    }

    obj = {
        "X_evo": X_evo,
        "X_struct": X_struct,
        "X_func": X_func,
        QICF_QUALITY_KEY: qicf_q,
        "y": y,
        "pairs": pairs[["human_uniprot", "virus_uniprot"]].copy(),
        "meta": meta,
    }

    joblib.dump(obj, out_path, compress=3)
    summary_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    stage_bar.update(1)
    stage_bar.set_postfix_str("saved")
    stage_bar.close()
    print(f"Saved: {out_path}")
    print(f"Saved summary: {summary_path}")
    print("Shapes:", X_evo.shape, X_struct.shape, X_func.shape, y.shape)
    print("QICF quality:", json.dumps(qicf_summary, ensure_ascii=False))
    print("Meta:", json.dumps(meta["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
