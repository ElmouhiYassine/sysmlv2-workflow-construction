# sysmlv2-workflow-construction

Python project to transform workflows into **SysML v2**, with three main sources:
- **Sap-SAM BPMN** (BPMN)
- **SOP-Bench** (SOPs)
- **WorFBench** (workflows)

## Repository structure

- `Sap-SAM BPMN/`
  - `bpmn_parser.py`: BPMN extraction (tasks, nodes, edges, pools, lanes)
  - `bpmn_into_sysml.py`: SysML generation from extracted BPMN
  - `run_transformation.py`: batch transformation of SAP-SAM CSV files
  - `inspect_data.py`: metrics and inspection of valid BPMN models
- `SOP-Bench/`
  - `main.py`: extraction, SysML transformation, CSV export, metrics
- `WorfBench/`
  - `worfbench.py`: plan extraction, SysML transformation, CSV export
  - `calculate_metrics.py`: metrics structures/printing helpers
- `Transformers/`
  - `graph_to_sysml.py`: generic graph-to-SysML transformer

## Installation

Recommended prerequisites:
- Python 3.10+
- virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

You can first try the transformation workflow in `Tansformation_guide.ipynb` to see how models are extracted and transformed step by step.

## Quick run

### 1) SAP-SAM BPMN

```powershell
python -u ".\Sap-SAM BPMN\run_transformation.py"
```

Output:
- `Sap-SAM BPMN/sapsam_sysml.csv`

### 2) SOP-Bench

```powershell
python -u ".\SOP-Bench\main.py"
```

Output:
- `SOP-Bench/sopbench_sysml.csv`

### 3) WorFBench

```powershell
python -u ".\WorfBench\worfbench.py"
```

Output:
- `WorfBench/worfbench_sysml.csv`

## Dataset Columns

Each dataset file contains the following columns:

| Column        | Description                                                                              |
|---------------|------------------------------------------------------------------------------------------|
| `origin`      | Encodes the provenance of the workflow sample. The format depends on the source dataset. |
| `user_prompt` | Natural language workflow description used as input for SysML v2 generation.             |
| `sysml_model`     | Transformed SysML v2 workflow model.                                                     |

### Origin Format

The `origin` column preserves traceability to the original dataset sample.

Examples:

| Dataset | Example | Meaning |
|---|---|---|
| SOPBench | `sopbench::train::42` | sample 42 from the train split |
| WorFBench | `worfbench::wikihow::test::15` | sample 15 from the `wikihow` source in the test split |
| SAP-SAM | `sapsam::1::192` | row 192 from CSV file `1.csv` |

## License

This project is licensed under the MIT License. full details in `LICENSE` file.
