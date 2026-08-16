# EnhTri-PHV

EnhTri-PHV predicts human-virus protein-protein interactions by combining
ProtT5 sequence representations, AlphaFold-derived structural descriptors,
Gene Ontology (GO)-based functional features, Quality-Informed Cross-Modal
Fusion (QICF), and a LightGBM classifier.

This repository is intended for academic research and reproduction of the
paper's **Benchmark experiment**. The runnable workflow is deliberately fixed
to the paper-default configuration: QICF `reliability_v2`, arithmetic-mean
pair-quality aggregation, the released pair-level 8:2 split, seed 42,
five-fold training, and CPU LightGBM. Ablation experiments and alternative
fusion, quality, and split settings are not included.

## Repository contents

```text
EnhTri-PHV/
|-- scripts/
|   |-- step0_build_benchmark_from_lstm_csv.py
|   |-- step1_prepare_data.py
|   |-- download_alphafold_mmcif.py
|   |-- step2_extract_features_joblib.py
|   `-- step3_train_lgbm.py
|-- data/
|   |-- base/lstm_phv_ppi_with_sequences.csv
|   |-- raw/Benchmark/
|   |-- processed/Benchmark/
|   `-- resources/
|-- models/
|-- outputs/
|-- results/Benchmark/
|-- requirements.txt
`-- SHA256SUMS.txt
```

The released Benchmark train/test files are frozen. The same protein may occur
in both partitions through different protein pairs, but identical labelled
protein pairs do not overlap. This is a pair-level Benchmark evaluation, not a
strict protein-disjoint generalization experiment.

## Data provenance

The Benchmark dataset was originally described by Tsukiyama et al.:

> Tsukiyama, S., Hasan, M. M., Fujii, S. & Kurata, H. LSTM-PHV: prediction of
> human-virus protein-protein interactions by LSTM with word2vec. *Briefings in
> Bioinformatics* **22**, bbab228 (2021).
> [https://doi.org/10.1093/bib/bbab228](https://doi.org/10.1093/bib/bbab228)

The exact sequence-associated CSV analysed here was downloaded on 30 January
2026 from the
[Google Drive collection](https://drive.google.com/drive/folders/1xF6MgGF5Ctfovg9KuSCTwp6C2P2os2Mj)
linked by
[`zelezniak-lab/PPI_prediction`](https://github.com/zelezniak-lab/PPI_prediction).
Its SHA256 checksum is:

```text
f562cb60a6637017657272211dff25fa05c62f249232ab8b27815225e9c753bf
```

The frozen files under `data/raw/Benchmark/` and
`data/processed/Benchmark/` are the inputs used by the main reproduction
workflow. The optional Step 0 reconstruction writes to
`data/raw/Benchmark_reconstructed/` and does not overwrite the released split.

## Environment

Python 3.10 is recommended. Create an environment and install the pinned
dependencies:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The verified software environment used Python 3.10.19, PyTorch 2.9.1+cu130,
Transformers 4.57.6, LightGBM 4.6.0, scikit-learn 1.7.2, NumPy 2.2.6,
pandas 2.3.3, Biopython 1.81, and joblib 1.2.0.

The reference workstation used Windows 11 Pro, an AMD Ryzen 5 9500F CPU,
32 GB RAM, and an NVIDIA GeForce RTX 5060 GPU with 8 GB VRAM. ProtT5 feature
extraction automatically uses CUDA when available and otherwise falls back to
the CPU. At repository preparation time, NVIDIA driver 610.62 reported CUDA
13.3, while the installed PyTorch build used its bundled CUDA 13.0 runtime. The
paper-default LightGBM stage is fixed to CPU execution.

## Required external resources

### ProtT5

Download `Rostlab/prot_t5_xl_half_uniref50-enc` into:

```text
models/prot_t5_xl_half_uniref50_enc/
```

For example:

```bash
hf download Rostlab/prot_t5_xl_half_uniref50-enc \
  --local-dir models/prot_t5_xl_half_uniref50_enc
```

### AlphaFold structures

Step 1 downloads available AlphaFold mmCIF files into:

```text
data/resources/alphafold_mmcif/
```

### GO resources

The frozen protein-to-GO table is included at:

```text
data/processed/Benchmark/uniprot_go.tsv
```

The included `data/resources/goa/goa_human.gaf.gz` is used in Step 2 to map GO
terms to the BP, MF, and CC namespaces. Live GO rebuilding is not part of the
released workflow because database updates could change the manuscript feature
snapshot.

ProtT5 weights and AlphaFold structures are not committed because of their
size and original distribution terms. Exact numerical reproduction requires
the same external resource snapshots and software versions; newer database
records may change feature coverage and small numerical details.

## Reproduction workflow

Run the following commands from the repository root. No experimental arguments
are required or exposed:

```bash
python scripts/step1_prepare_data.py
python scripts/step2_extract_features_joblib.py
python scripts/step3_train_lgbm.py
```

### Step 1: prepare the frozen Benchmark inputs

Implementation steps:

1. Read the released sequence file and the complete labelled, training, and
   test pair files from `data/raw/Benchmark/`.
2. Remove duplicate labelled pair rows and verify that the training and test
   pair keys do not overlap and jointly cover the complete Benchmark table.
3. Standardize UniProt identifiers, remove isoform suffixes for canonical
   resource lookup, and construct the protein table without changing the
   released pair split.
4. Write deterministic processed tables to `data/processed/Benchmark/`,
   including `pairs_all.csv`, `pairs_train.csv`, `pairs_test.csv`,
   `pairs_pos.csv`, `pairs_neg.csv`, `proteins.csv`, and
   `dataset_summary.txt`.
5. Preserve the released `uniprot_go.tsv` and download missing AlphaFold mmCIF
   structures into `data/resources/alphafold_mmcif/`.

Expected fixed counts are 246,213 total pairs, 22,383 positive pairs, 223,830
negative pairs, 196,970 training pairs, and 49,243 independent-test pairs.

### Step 2: extract trimodal features and QICF quality factors

Implementation steps:

1. Load the processed pair/protein tables, frozen GO annotations, GO namespace
   mapping, AlphaFold structures, and local ProtT5 model.
2. Normalize amino-acid sequences, truncate them to 512 tokens, obtain ProtT5
   residue representations, and apply mask-aware mean pooling to form
   protein-level sequence embeddings.
3. Construct role-preserving sequence pair features from the human and viral
   embeddings, absolute difference, element-wise product, squared difference,
   normalized product, and five global similarity/distance statistics.
4. Extract AlphaFold C-alpha structural descriptors using an 8 A contact
   threshold, at most 1,024 C-alpha atoms, the top 64 contact degrees, global
   geometry, hydrophobicity, coordinate coverage, and pLDDT statistics.
5. Construct the 40-dimensional GO semantic feature vector from the complete,
   BP, MF, and CC spaces using term counts, information content, shared terms,
   weighted Jaccard similarity, and weighted overlap similarity.
6. Calculate the fixed `reliability_v2` quality vector
   `[q_e, q_s, q_f]`. Sequence quality is fixed to one; structural and
   functional protein qualities follow the paper equations; human and viral
   protein qualities are combined only by arithmetic mean.
7. Save the aligned sequence, structure, function, label, pair, metadata, and
   QICF-quality arrays to
   `outputs/features/Benchmark/features.joblib`. ProtT5 and structural caches
   are stored in the same directory for resumable execution.

The feature payload can exceed 5 GB and is excluded from Git.

### Step 3: train QICF-LightGBM and evaluate the holdout set

Implementation steps:

1. Load the Step 2 payload and reject it unless the quality metadata specifies
   `reliability_v2`, arithmetic-mean aggregation, and the fixed paper constants.
2. Align feature rows with the released `pairs_train.csv` and
   `pairs_test.csv`, then reproduce the frozen pair-level 8:2 split.
3. Create five stratified folds inside the training partition using seed 42.
   The independent test set is never used to fit projections, select boosting
   iterations, or tune hyperparameters.
4. Within each fold, fit separate 64-dimensional Gaussian random projections
   for sequence, structure, and function using only that fold's fitting rows,
   then apply row-wise L2 normalization.
5. Assemble the fixed QICF representation from the original trimodal features,
   reliability-weighted projections, three element-wise cross-modal
   interactions, and the three quality factors.
6. Train CPU LightGBM with fold-specific class weighting, up to 8,000 boosting
   rounds, and early stopping after 300 rounds without validation-AUC
   improvement. The main parameters are learning rate 0.02, 255 leaves,
   feature fraction 0.85, bagging fraction 0.90, minimum 60 samples per leaf,
   and L2 regularization 3.0.
7. Average the five fold-model probabilities on the independent test set and
   compute AUROC, AUPRC, accuracy, precision, recall, F1, and MCC at threshold
   0.5.
8. Write the split table, metadata, per-fold report, out-of-fold metrics, and
   independent-test report to `outputs/Benchmark/`.

## Reference result

The archived seed-42 five-fold ensemble result on the released independent
test set is provided in `results/Benchmark/`:

| Metric | Value |
|---|---:|
| AUROC | 0.998562 |
| AUPRC | 0.994288 |
| Accuracy | 0.996893 |
| F1 | 0.982620 |
| MCC | 0.981088 |

## Integrity check

Verify the released files before running the workflow:

```bash
sha256sum -c SHA256SUMS.txt
```

On Windows PowerShell, use `Get-FileHash` to calculate SHA256 hashes.

## Citation

If this repository contributes to your research, please cite the associated
EnhTri-PHV paper. This is an academic citation request, not an additional
condition of the software license. Citation metadata are provided in
`CITATION.cff` and should be updated after a final DOI is assigned.

## License and third-party terms

The original source code is released under the BSD 3-Clause License; see
`LICENSE`. Benchmark data, ProtT5 weights, AlphaFold structures, GO resources,
and other third-party materials remain subject to the licenses and terms of
their original providers. The BSD 3-Clause License does not relicense those
third-party resources.
