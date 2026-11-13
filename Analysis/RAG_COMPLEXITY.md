# Quality-first, complexity-aware RAG corpus

`python -m Analysis.normalize_rag_pool` normalizes every eligible train-side candidate and validates
the result with the existing SysML v2 kernel loaded with the MOSAICO library. The command maintains
`enriched_dataset_analysis/rag/normalization_state_v2.sqlite`; rerunning it skips origins already
processed. Kernel requests validate independent package-scoped models in sequential batches; a
failing or oversized batch is bisected until it fits or the individual invalid workflow and its
exact error are isolated. The shared kernel is restarted every 2,500 pending workflows to bound
session growth. Temporary unique package names are used only in batched parser requests and are
never written to the saved normalized model. Rows that cannot be normalized without guessing are retained in
`normalization_results.csv` with a failure reason and are excluded from selection.
Use `python -m Analysis.normalize_rag_pool --retry-failures` after a normalization fix to retry only
previously flagged or parser-invalid origins.

Parser validity and structural completeness are separate gates. Structural validation requires
executable control flow, an outgoing entry edge from `start`, all declared actions reachable from
`start`, a reachable `done`, and a path from each reachable action to `done`. Defects are recorded
without adding or inferring edges. To document an intentional exception, add a source comment such
as `// RAG-STRUCTURAL-EXCEPTION: reason`; it is reported distinctly but remains excluded from RAG.
Only candidates passing both the parser and structural checks are selected.

The normalizer preserves the original graph topology and source action descriptions, outputs,
agent types and valid assignments. It adds the required MOSAICO and ScalarValues imports, assigns
`Step` to actions, writes descriptions using `::>`, adds a default `solutionAgent` only when needed,
and rewrites declarations before control flow. Untyped outputs are inferred as Boolean only when
used as Boolean decision guards or when the description explicitly says the output is true/false;
all other untyped outputs are flagged for review. WorFBench's `then actionX;` references after a
fork remain branch targets, not sequential actions.

For complexity classification, each candidate feature is mapped to its empirical percentile rank
in the quality-qualified train pool (range 0–1). The score is the weighted sum:

| Feature | Weight |
|---|---:|
| Actions | 0.08 |
| Decisions | 0.18 |
| Forks | 0.15 |
| Joins | 0.15 |
| Control-flow edges | 0.10 |
| Conditional branches | 0.12 |
| Parallel branches | 0.08 |
| Nested parallelism depth | 0.10 |
| Cycles | 0.02 |
| Repeated action occurrences | 0.02 |

The score's empirical one-third and two-thirds quantiles are saved in `rag_selection_config.json`;
values through the first threshold are **Simple**, through the second are **Medium**, and the rest
are **Complex**. Maximum execution-path depth is recorded when the graph is acyclic but is not in
the score. Nested parallelism is measured by the deepest fork containment before the nearest
reachable common declared join; without such a join, nesting is not asserted. Parallel-branch count
is the total outgoing branches at forks with at least two successors; conditional-branch count
includes each `if` and optional `else`. Cycles are detected on the directed control-flow graph.
Boolean `has_parallelism` and `has_nested_parallelism` flags accompany their branch-count and depth
features. Repeated action occurrences count action nodes on a directed cycle; multiple incoming
edges alone do not count as repetition. Duplicate action bodies are tracked separately in
`duplicate_action_declarations`, and `max_incoming_edges` is retained as additional metadata.

The selector now admits **up to 10** examples per existing category, including zero. There are no
3/4/3 quotas, minimum category sizes, structural replacements, or relaxed-filter fallbacks.
`Analysis/rag_quality.py` applies the hard admission rules before clustering/ranking:

| Configurable option | Default hard limit |
|---|---:|
| `--max-actions` | 15 |
| `--max-decisions` | 5 |
| `--max-forks` | 2 |
| `--max-joins` | 2 |
| `--max-description-chars` | 1500 |
| `--max-per-category` | 10 (upper cap only; configurable from 1 to 10) |

Parser evidence must be present and successful in `normalization_results.csv`; absent/unassessed
results fail closed, not as syntax-valid. Selection never starts a kernel or changes the saved
parser flags. Static checks read the current normalized text again and require canonical imports,
Step declarations, descriptions, declared agents, typed outputs and resolvable guards. The
normalizer is used only as a non-mutating consistency check; models requiring further repairs are
excluded, not rewritten. Structural checks require action and control-node reachability and
termination paths, one entry edge, no edges back to `start`, deterministic action successors,
nondegenerate forks/joins/merges, complete decision blocks and guard outputs available on every
path to their decision. Parallel regions must have a unique common synchronization join, no
overlapping branches before that join, and no branch bypassing it to terminate. These conservative
rules may reject unusual but intentional workflows. Cycles through decisions remain eligible;
unguarded self-loops and unexplained cycles are rejected.

Description quality uses deterministic **lexical proxies, not semantic-equivalence certification**.
Descriptions need at least eight words and no placeholder/explicitly ambiguous text. Camel-case
action names and action descriptions are compared with narrative content words (English
stopwords/template words removed; simple plural normalization). By default, at least 80% of
informative actions must have at least 30% token overlap with the narrative
(`--min-action-coverage`, `--min-action-token-overlap`). Parallel/cyclic graphs also need explicit
parallel/repetition language. This does not prove ordering, guard polarity or arbitrary prose
faithfulness; lexical paraphrases can be false negatives. No LLM, new embedding, external service
or extra dependency is used. The lexical quality score is 0.7 × action coverage + 0.3 × mean
action-token overlap.

Candidates are ranked lexicographically: quality bands of 0.05, description-length bands of 250
characters, action-count bands of five, then marginal construct coverage, semantic-cluster novelty,
centroid similarity, exact description length and source ID. This treats nearly equal quality and
length as ties so diversity remains useful, but never lets diversity override a better quality
or conciseness band. Construct coverage includes sequences, decisions, loops, parallelism,
synchronization, repeated execution occurrences and nested parallelism. Complexity labels are
metadata only. Semantic near-duplicates (cosine ≥ 0.90 by default) never fill vacant slots.
Train/test exclusions and the description-embedding train-to-test leakage guard are unchanged;
no held-out model text or structural features enter filtering, ranking or threshold fitting.

Run, for example:

```sh
python -m Analysis.rag_selection --strategy complexity \
  --max-actions 15 --max-decisions 5 --max-forks 2 --max-joins 2 \
  --max-description-chars 1500 --max-per-category 10
```

The selected CSV retains the retrieval columns and adds quality scores/coverage metadata.
`rag_selection_summary.csv` logs candidates, qualifying, selected and rejected counts for **all**
categories. `rag_candidate_audit.csv` records every train candidate's filter/ranking outcome;
`rag_rejection_reasons.csv` aggregates reasons by category. Reason counts may overlap: a candidate
can fail multiple filters. `near_duplicate` and `category_limit` distinguish ranking exclusions
from failed quality admission. Empty categories retain summary rows and figures; singleton
figures use zero coordinates instead of invalid two-component PCA.

The original 130-example semantic corpus and its configuration are preserved in
`enriched_dataset_analysis/rag/baseline/` and is not overwritten. `--strategy semantic` now applies
the same mandatory quality gates and omits only the construct-diversity tie-break; the baseline
files, not that flag, preserve the historical centroid-only corpus. Run the new selection with
`python -m Analysis.rag_selection --strategy complexity`. Generate the old/new comparison and
excluded-origin report with `python -m Analysis.compare_rag_corpora`. It writes the corpus
comparison, source-dataset validation counts, and excluded origins with their reasons.

The required `::>` description operator is accepted by the local SysML kernel and is used in the
normalized corpus as requested. In KerML, `::>` is formally a references operator; retain this
project-specific form only while it remains compatible with the installed parser and MOSAICO
library.
