# SysML v2 Workflow Dataset

This repository contains scripts and research artifacts for transforming workflow datasets into
SysML v2 action flows, validating the generated models, and studying workflow semantics with
sentence embeddings. Transformation and parser behavior are kept in the source-specific code; the
analysis scripts operate on the resulting datasets.

## Dataset sources

- **SOPBench** — `Zekunli/SOPBench` on Hugging Face.
- **WorFBench** — `zjunlp/WorFBench_train` and `zjunlp/WorFBench_test` on Hugging Face.
- **SAP-SAM** — [Zenodo record 7012043](https://zenodo.org/records/7012043), containing BPMN models.

The generated dataset CSVs use `origin`, `user_prompt`, and `sysml_model` columns. Enriched
datasets additionally contain `detailed_description`. `origin` keeps the source workflow ID, for
example `sopbench::train::42`, `worfbench::wikihow::test::15`, or `sapsam::1::192`.

Large original datasets and local research outputs are retained in the workspace but are not
required to install the code. SAP-SAM's raw collection is especially large. Obtain source data from
its upstream provider when it is not available locally; do not commit credentials or local
environment files.

## Installation

Use Python 3.10 or newer from the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. Install the analysis and test dependencies
when reproducing embedding studies or running the full test suite:

```sh
python -m pip install -r requirements-analysis.txt
# or install analysis dependencies plus tests:
python -m pip install -r requirements-dev.txt
```

The SysML kernel validator also requires the external Jupyter `sysml` kernel and
`MOSAICO.sysml` library used by the original parser project. Configure their locations without
editing source code:

```sh
export SYSML_MOSAICO_PATH=/path/to/Specialized_sysml_generator/library/MOSAICO.sysml
# Optional when the kernel is not in Jupyter's normal kernelspec search locations:
export SYSML_KERNEL_DIR=/path/to/jupyter/kernels
```

The validator defaults to the `Specialized_sysml_generator` sibling directory and Jupyter's
standard kernel search path. It considers warnings acceptable; parser `ERROR:` output or a
Jupyter execution error makes a model invalid.

## Project map

| Path | Purpose |
|---|---|
| `SAP_SAM_BPMN/` | SAP-SAM BPMN loading, filtering, BPMN parsing, and SysML v2 transformation. |
| `SOPBench/` | SOPBench graph extraction and SysML v2 transformation. |
| `WorfBench/` | WorFBench source handling, graph extraction, SysML v2 transformation, metrics, and generated dataset CSVs. |
| `Transformers/` | Shared graph-to-SysML transformer used by WorFBench. |
| `update_datasets.py` | Combines the three source-specific SysML CSVs into one dataset. |
| `validate_sysml_datasets.py` | Resumable SysML kernel validation and parser error reporting. |
| `enriched dataset/` | Original enriched CSVs and corrected copies under `fixed/`; originals are not overwritten. |
| `Analysis/` | Dataset analysis, embedding/clustering studies, quality checks, and RAG selection. |
| `enriched_dataset_analysis/` | Saved analysis tables, figures, assignments, embedding caches, and RAG results. |
| `rag_experiments/` | Reproducible corpus extraction, stratified sets, verification, and their outputs. |
| `tests/` | Unit tests for SysML workflow checks, RAG selection, and normalization. |
| `Tansformation_guide.ipynb` | Exploratory notebook for inspecting and transforming sample workflows. |

Research outputs and input datasets are kept separate from the transformation scripts. The
`enriched dataset/` CSVs are the untouched source inputs for semantic analysis; corrected copies
and their fix scripts are kept together under `enriched dataset/fixed/`.

### Script and artifact inventory

- **SAP-SAM:** `download_sapsam_data.py` retrieves the upstream archive;
  `bpmn_parser.py` extracts BPMN structures; `bpmn_into_sysml.py` generates SysML;
  `run_transformation.py` performs the indexed batch conversion; `inspect_data.py` reports on raw
  BPMN content; `model_description.py` contains optional BPMN element-description utilities;
  `get_clean_json.py` is a one-record JSON inspection helper.
- **SOPBench:** `run_transformation.py` builds workflow graphs and writes SysML;
  `inspect.py` displays one source example. `trash.py` is a retained alternate/older copy of the
  transformation script and is not part of the documented pipeline; review before removing it.
- **WorFBench:** `run_transformation.py` extracts and transforms source workflows;
  `calculate_metrics.py` contains batch counters; `analysis.py` is a second standalone analysis/
  transformation implementation; `read_the_data.py` prints a local CSV for manual inspection.
  Both transformation scripts write to `WorfBench/worfbench_dataset/`, regardless of the current
  working directory. The older differing export is retained in that directory's `legacy/` folder.
- **`Analysis/`:** `analyze_dataset_embeddings.py` is the earlier embedding/structure study;
  `enriched_pipeline.py` runs the staged enriched-data embedding and clustering workflow;
  `enriched_report.py` renders reports and figures; `sysml_workflow.py` parses supported workflow
  syntax and computes structural features; `rag_quality.py`, `rag_selection.py`, and
  `normalize_rag_pool.py` provide quality checks, selection, and resumable normalization/parser
  validation; `compare_rag_corpora.py` compares candidate corpora; `extract.py` prepares the
  WorFBench software-engineering subset for annotation.
- **`enriched dataset/fixed/`:** the three `fix_*.py` scripts write corrected copies of enriched CSVs;
  `validate_csv.py` is a convenience wrapper around the shared kernel validator. Original files
  remain in the parent directory, while `history/` contains the retained SOPBench before/after files.
- **Generated results:** `Analysis/dataset_analysis/` contains the earlier embedding study;
  `enriched_dataset_analysis/` contains the full clustering run, category assignments, reports,
  figures, and embedding caches; `validation_results_enriched/` contains parser-validation records;
  `rag_experiments/data/` contains selected corpora, manifests, statistics, audits, and resumable
  extraction state. These outputs are kept because they document existing research results.

## Reproducing dataset construction

### SAP-SAM

If the raw data is not present, download and extract it from the repository root:

```sh
python -m SAP_SAM_BPMN.download_sapsam_data
```

Run the batch transformation from the repository root. The command uses the downloaded models
and precomputed index under `SAP_SAM_BPMN/sap_sam_2022/`, and writes
`SAP_SAM_BPMN/sap_sam_sysmlv2/sapsam_sysmlv2.csv`:

```sh
python -m SAP_SAM_BPMN.run_transformation
```

### SOPBench

```sh
python SOPBench/run_transformation.py
```

### WorFBench

```sh
python -m WorfBench.run_transformation
```

These source loaders use Hugging Face `datasets`; the first run may download upstream data. To
combine existing source-specific CSVs, use the repository-relative defaults or provide explicit
paths:

```sh
python update_datasets.py
python update_datasets.py --sopbench path/to/sopbench.csv \
  --worfbench path/to/worfbench.csv --sapsam path/to/sapsam.csv \
  --output path/to/unified_sysmlv2.csv
```

## Validation and quality checks

`validate_sysml_datasets.py` reuses one SysML kernel session, records exact parser errors, accepts
warnings, and checkpoints results so interrupted runs can resume. Its default input is the
SAP-SAM enriched CSV under `enriched dataset/fixed/`; its default output is `validation_results_enriched/`.
Run a short smoke test or the configured full validation:

```sh
python validate_sysml_datasets.py --smoke-test
python validate_sysml_datasets.py
```

Set `SYSML_MOSAICO_PATH` and, if needed, `SYSML_KERNEL_DIR` as described above before running.
`enriched dataset/fixed/fix_sopbench.py`, `fix_worfbench.py`, and `fix_sapsam.py` write corrected
copies under `enriched dataset/fixed/`; the source files in `enriched dataset/` are left untouched.

## Analysis and embedding studies

The main enriched-dataset pipeline has separate, resumable stages. Run from the repository root:

```sh
python -m Analysis.enriched_pipeline prepare
python -m Analysis.enriched_pipeline embed
python -m Analysis.enriched_pipeline sweep
python -m Analysis.enriched_pipeline cluster
python -m Analysis.enriched_pipeline compare
python -m Analysis.enriched_report
```

The default configuration reads the three files in `enriched dataset/fixed/` and writes tables, figures,
embedding caches, cluster assignments, and reports to `enriched_dataset_analysis/`. Configuration
and random seed are saved with the experiment outputs.

The earlier source-dataset embedding and structural-analysis study reads CSVs relative to the
`Analysis/` directory and writes `Analysis/dataset_analysis/`:

```sh
cd Analysis
python analyze_dataset_embeddings.py
cd ..
```

The quality-first RAG selector and corpus-comparison utilities are in `Analysis/rag_selection.py`
and `Analysis/compare_rag_corpora.py`. The larger fixed-corpus and stratified experiments, including
their eligibility rules and output schemas, are documented in [`rag_experiments/README.md`](rag_experiments/README.md).
Those experiments reuse saved category assignments and embeddings and require the configured
SysML kernel for final parser validation.

To reproduce the validated RAG pool and category selection after running the embedding pipeline:

```sh
python -m Analysis.normalize_rag_pool
python -m Analysis.rag_selection
```

## Tests

From the repository root:

```sh
python -m pytest -q
python -m compileall -q Analysis SAP_SAM_BPMN SOPBench Transformers WorfBench \
  rag_experiments tests update_datasets.py validate_sysml_datasets.py
```

The automated tests cover deterministic structural/quality selection and normalization helpers.
Full transformation, embedding, and parser-validation runs additionally require the relevant
upstream datasets, analysis inputs, model downloads, and external SysML kernel.

## License

See [`LICENSE`](LICENSE).
