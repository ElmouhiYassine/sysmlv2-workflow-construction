# Fixed 2,000-workflow retrieval corpus

From the repository root, using the existing project environment:

```sh
python -m rag_experiments.extract_corpus
```

Only `enriched dataset/worfbench_nemotron_together.csv` and
`enriched dataset/sapsam_nemotron_together.csv` supply examples. The `origin`,
`detailed_description` and `sysml_model` mappings come from the existing utilities.
Output descriptions and code are the **exact original CSV strings**, not normalized replacements.
No generation, enrichment, external dataset fetches or LLM calls occur.

The script imports the existing subset reader, graph checks, quality checks, complexity scoring,
cached MiniLM description embeddings and SysML kernel helpers. A canonical in-memory view is used
only to inspect legacy syntax safely; its topology must match the original. It is never exported.
Missing or inconsistent original references, duplicate declarations, incomplete graphs, suspicious
branching/synchronization and poor lexical narrative coverage fail admission. Lexical alignment
is a proxy, not a proof of semantic equivalence.

Unlike the earlier concise 130-example corpus, this experiment permits larger examples:
defaults are at most 60 actions, 20 decisions, 10 forks, 10 joins and 12,000 description characters.
These are configurable hard limits; no criteria are relaxed to fill the corpus. Size bands are
small (1–5 actions), medium (6–15) and large (16+). Complexity levels use the existing empirical
percentile formula, fitted only on qualified training candidates.

WorFBench native scenario/source labels are recovered from the original IDs using the existing
`get_source_from_origin` utility. `category` for **both sources** is the existing category-analysis assignment from
`enriched_dataset_analysis/workflow_assignments.csv`, also retained in `analysis_category`
for compatibility. Native WorFBench source labels are stored separately in `scenario`.
SAP-SAM has no native scenario labels, and neither CSV provides original domain labels:
those fields remain **null**.

Sampling water-fills sources and native scenarios (or existing analysis categories when native
metadata is absent), then size/structure/complexity cells. Quality orders candidates within each
cell; seed 42 resolves ties. This balances both sources without requiring impossible equal
category quotas. Structural families include sequential, conditional, looping, parallel, nested
parallel and combined decision/parallel graphs.

Explicit test IDs and `--exclude-ids` never enter graph assessment or selection. Existing cached
description embeddings exclude training neighbours of held-out descriptions at cosine >= 0.95.
The cache is accepted only when its mapped description matches the source after the existing
whitespace/markup normalization. No held-out gold model guides ranking or structural diversity.

Deduplication rejects normalized identical descriptions, exact code and identifier-independent
labelled-graph duplicates; greedy selection rejects cosine >= 0.90 to any already selected
description. The final full pairwise similarity matrix is verified. This is a declared operational
definition of near-duplication, not a guarantee against every possible semantic paraphrase.

Every selected **original** SysML text is then checked by the existing MOSAICO-loaded SysML kernel,
sequentially in batches. The reused batch helper temporarily aliases independent package names
in kernel requests only and bisects errors to isolate invalid workflows. Warnings are acceptable;
`ERROR:` lines or Jupyter errors invalidate a workflow. Invalid candidates are excluded and
replacements come through the same admission and diversity rules.

Quality and parser checkpoints are bound to exact text and implementation/kernel hashes in
`data/extraction_state.sqlite`. Restarting the command resumes cached work without accepting
stale results. Deterministic selection is replayed after parser exclusions; serialized source
texts are checked against the loaded CSV strings.

Outputs:

- `data/rag_corpus_2000.jsonl`: original pairs, IDs, original metadata, separate derived metadata,
  structural features and parser validity.
- `data/rag_corpus_manifest.json`: input/code/kernel hashes, exact ordered IDs, sampling settings
  and limitations, reproducibility checks and final corpus checksum.
- `data/rag_corpus_statistics.json`: sources, categories/scenarios/domains, sizes, action counts,
  structures, complexity, duplicate exclusions and parser outcomes.
- `data/candidate_audit.jsonl`: quality and duplicate rejection audit.
- `data/category_distribution.csv`: source/scenario and derived-category count table.

The original WorFBench scenarios `alfworld`, `intercodesql`, `os`, `seal_tools` and `webshop`
occur only in test rows in these input CSVs. They are reported as unavailable rather than used
to fill coverage. `environment/os` and `environment/webshop` are distinct native training labels.

If 2,000 qualifying distinct examples cannot be obtained, the script writes the smaller corpus
and shortage statistics and exits with status 2. It never pads with invalid or near-duplicate data.
All scripts, checkpoints and outputs created by this experiment remain under `rag_experiments/`.

Tests:

```sh
python -m pytest -q rag_experiments/test_extract_corpus.py
python -m rag_experiments.verify_corpus
```

## Separate sequential and complex stratified sets

```sh
python -m rag_experiments.extract_stratified_1000
```

This command reads the full enriched WorFBench/SAP-SAM source CSVs, not the fixed 2,000-example
corpus. It writes `data/rag_sequential_500.csv` and `data/rag_complex_500.csv` without changing
existing datasets or RAG configurations. Original `origin` and `user_prompt` values are preserved;
`sysml_model` is exported unchanged under the requested `sysml_code` name. Original enriched
descriptions are retained separately in `description`, because many SAP-SAM `user_prompt` fields
are empty. Category labels come from the existing category analysis.

Sequential eligibility requires 1–8 actions, no decisions/forks/joins and no cycles. Complex
eligibility requires 1–8 actions, 1–2 decisions **and** 1–2 forks, with correct synchronization
checked by the existing graph-quality functions. Both sets use the established concise-description
limit of 1,500 characters and conservative lexical/graph quality checks; placeholder action names
are excluded. All final original models must pass the shared kernel. Saved parser evidence is
reused only when its exact model and kernel-context hashes match, through a read-only connection
to the previous journal. New validation state is stored separately.

For each set, category soft targets are `category_eligible / total_eligible * 500`.
The denominator is the static quality-qualified original pool after known parser exclusions,
before selected-set deduplication; unselected candidates are not all kernel-assessed.
Largest proportional deficit determines the next category, with quality ranking and seed-42 tie
ordering within categories. Exact descriptions/code/labelled graphs and description cosine
>= 0.90 prevent duplicate selection. Targets never override these filters. A shortage produces
a smaller CSV and explicit counts rather than padding. Test workflows and held-out neighbours
remain excluded.

`data/rag_stratified_category_allocations.csv` includes every category's eligible count, fractional
target, actual selection and difference, including zero-eligibility categories.
`data/rag_stratified_1000_summary.json` records source/structural counts, shortages, parser validity,
settings, hashes and reproducibility evidence.
