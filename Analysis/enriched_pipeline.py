"""Semantic clustering and characterisation of the enriched SysML v2 dataset.

Stages (run from the project root):

    python -m Analysis.enriched_pipeline prepare     # load, quality checks, descriptive statistics
    python -m Analysis.enriched_pipeline embed       # cached sentence embeddings (long-text chunking)
    python -m Analysis.enriched_pipeline sweep       # candidate-K evaluation
    python -m Analysis.enriched_pipeline cluster     # final clustering + interpretation tables
    python -m Analysis.enriched_pipeline compare     # MiniLM vs MPNet stability on a shared sample
    python -m Analysis.enriched_report                # categories, figures, report (no recomputation)

Only the enriched natural-language description is embedded. SysML models are used only for
structural statistics. The enriched CSV files are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from Analysis.analyze_dataset_embeddings import (
    get_dataset_from_origin,
    get_source_from_origin,
    nearest_neighbor_distances,
)

MODEL_ALIASES = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}

# Explicit, configurable mapping from logical fields to the physical enriched-CSV columns.
COLUMN_MAP = {
    "origin": "origin",
    "description": "detailed_description",
    "original_prompt": "user_prompt",
    "sysml": "sysml_model",
}

# Words that describe the narrative template rather than the task. Used ONLY to make
# keyword tables readable; embeddings are computed on the untouched description.
TEMPLATE_WORDS = {
    "process", "begins", "begin", "starts", "start", "following", "follows", "followed", "then",
    "next", "finally", "action", "actions", "step", "steps", "workflow", "completes", "completed",
    "completion", "performed", "performs", "perform", "executed", "execution", "path", "branch",
    "branches", "flow", "flows", "parallel", "decision", "determines", "checks", "checked",
    "proceeds", "proceed", "continues", "continue", "ends", "end", "finishes", "finish", "occur",
    "occurs", "occurring", "simultaneously", "concurrently", "result", "results", "outcome",
    "depending", "based", "initiates", "initiated", "triggers", "triggered", "leads", "lead",
    "case", "otherwise", "condition", "conditions", "met", "using", "used",
}


@dataclass
class Config:
    input_dir: str = "enriched dataset/fixed"
    output_dir: str = "enriched_dataset_analysis"
    input_files: dict = field(default_factory=lambda: {
        "sopbench": "sopbench_nemotron_together.csv",
        "worfbench": "worfbench_nemotron_together.csv",
        "sapsam": "sapsam_nemotron_together.csv",
    })
    column_map: dict = field(default_factory=lambda: dict(COLUMN_MAP))
    min_words: int = 8
    model: str = "minilm"
    batch_size: int = 64
    long_text: str = "chunk"          # "chunk" (overlapping windows, pooled) or "truncate"
    chunk_overlap_tokens: int = 64
    reuse_cache: bool = True
    sample_size: int = 0              # 0 = all workflows (used for MPNet comparison runs)
    k_min: int = 5
    k_max: int = 20
    k: int = 12
    sweep_seeds: tuple = (0, 1, 2)
    silhouette_sample: int = 10000
    algorithm: str = "kmeans"         # "kmeans" or "minibatch"
    n_init: int = 10
    seed: int = 42
    n_representative: int = 10
    n_random_examples: int = 10
    n_validation_per_category: int = 20
    near_duplicate_threshold: float = 0.95


# ---------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------
class Paths:
    def __init__(self, cfg: Config):
        self.root = Path(cfg.output_dir)
        self.figures = self.root / "figures"
        self.tables = self.root / "tables"
        self.embeddings = self.root / "embeddings"
        for d in (self.root, self.figures, self.tables, self.embeddings):
            d.mkdir(parents=True, exist_ok=True)
        self.model_tag = "mpnet" if "mpnet" in resolve_model(cfg.model) else "minilm"
        if cfg.sample_size:
            self.model_tag += f"_sample{cfg.sample_size}"
        self.emb_file = self.embeddings / f"embeddings_{self.model_tag}.npy"
        self.emb_meta = self.embeddings / f"embedding_metadata_{self.model_tag}.json"
        self.emb_index = self.embeddings / f"embedding_index_{self.model_tag}.csv"
        self.base = self.root / "workflows_base.parquet"
        self.assignments = self.root / "workflow_assignments.csv"
        self.clusters = self.root / f"cluster_labels_{self.model_tag}.npy"


def resolve_model(name: str) -> str:
    return MODEL_ALIASES.get(name.lower(), name)


# ---------------------------------------------------------------------------------------
# Data loading and quality checks
# ---------------------------------------------------------------------------------------
def clean_column_name(name: str) -> str:
    return name.replace("\ufeff", "").strip().strip('"').strip()


def normalize_description(text: str) -> str:
    """Whitespace/markup normalisation only. Wording, negation and terminology are kept."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"[*_`#]{1,3}", " ", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip()


_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
_FIRST_THEN_RE = re.compile(r"\bfirst\s+(\w+)\s+then\s+(\w+)")


def structure_features(sysml: str) -> dict:
    """Regex-based control-flow counts on the generated (regular, machine-written) SysML.

    Limitations: this is not a SysML parser. String literals are removed first so that
    descriptions cannot inflate the counts. Synthetic helper actions named ``conditionN`` are
    excluded from ``num_actions``. Anonymous ``then fork;`` and named ``fork x;`` are both
    counted. Iteration is detected only through explicit ``first A then B`` edges (cycle
    search); chained ``then`` sequences used by WorFBench cannot express loops, so for that
    dataset ``has_iteration`` is a lower bound that is in practice always False.
    """
    code = _STRING_RE.sub('""', sysml or "")
    actions = [m for m in re.findall(r"\baction\s+(?!def\b)(\w+)", code) if not re.fullmatch(r"condition\d+", m)]
    n_actions = len(actions)
    n_decisions = len(re.findall(r"\bdecide\s+\w+", code))
    n_forks = len(re.findall(r"\bfork\b(?!\s*\w*\s*:)", code))
    n_joins = len(re.findall(r"\bjoin\s+\w+", code))

    graph: dict[str, set] = {}
    for a, b in _FIRST_THEN_RE.findall(code):
        graph.setdefault(a, set()).add(b)
    return {
        "num_actions": n_actions,
        "num_decisions": n_decisions,
        "num_forks": n_forks,
        "num_joins": n_joins,
        "has_iteration": _has_cycle(graph),
    }


def _has_cycle(graph: dict) -> bool:
    color: dict[str, int] = {}
    for root in graph:
        if color.get(root):
            continue
        stack = [(root, iter(graph.get(root, ())))]
        color[root] = 1
        while stack:
            node, it = stack[-1]
            for nxt in it:
                state = color.get(nxt, 0)
                if state == 1:
                    return True
                if state == 0:
                    color[nxt] = 1
                    stack.append((nxt, iter(graph.get(nxt, ()))))
                    break
            else:
                color[node] = 2
                stack.pop()
    return False


def load_enriched(cfg: Config) -> tuple[pd.DataFrame, dict]:
    cmap = cfg.column_map
    frames, report = [], {"files": {}, "malformed_records": []}
    for key, filename in cfg.input_files.items():
        path = Path(cfg.input_dir) / filename
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        df.columns = [clean_column_name(c) for c in df.columns]
        missing_cols = [c for c in (cmap["origin"], cmap["description"], cmap["sysml"]) if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{path}: expected columns {missing_cols}; found {list(df.columns)}")
        out = pd.DataFrame({
            "origin": df[cmap["origin"]].str.strip(),
            "raw_description": df[cmap["description"]],
            "original_prompt": df[cmap["original_prompt"]] if cmap["original_prompt"] in df.columns else "",
            "sysml_model": df[cmap["sysml"]],
            "input_file": filename,
        })
        report["files"][key] = {"path": str(path), "rows": len(out), "columns": list(df.columns)}
        frames.append(out)
    data = pd.concat(frames, ignore_index=True)

    data["dataset"] = data["origin"].map(get_dataset_from_origin)
    data["source"] = data["origin"].map(get_source_from_origin)
    data["description"] = data["raw_description"].map(normalize_description)
    data["n_words"] = data["description"].str.split().str.len()
    data["n_sentences"] = data["description"].map(lambda t: len([s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]))
    data["original_prompt_words"] = data["original_prompt"].str.split().str.len()
    data["sysml_words"] = data["sysml_model"].str.split().str.len()

    data["missing_description"] = data["raw_description"].str.strip() == ""
    data["too_short"] = (~data["missing_description"]) & (data["n_words"] < cfg.min_words)
    data["malformed"] = (data["origin"] == "") | (data["dataset"] == "unknown") | (data["sysml_model"].str.strip() == "")
    data["duplicate_origin"] = data["origin"].duplicated(keep=False) & (data["origin"] != "")
    norm = data["description"].str.lower()
    data["duplicate_description"] = norm.duplicated(keep="first") & ~data["missing_description"]
    data["in_duplicate_group"] = norm.duplicated(keep=False) & ~data["missing_description"]
    data["used_in_clustering"] = ~(data["missing_description"] | data["too_short"])

    feats = pd.DataFrame([structure_features(s) for s in data["sysml_model"]])
    data = pd.concat([data, feats], axis=1)
    data["has_branching"] = data["num_decisions"] > 0
    data["has_parallelism"] = (data["num_forks"] > 0) | (data["num_joins"] > 0)
    data["workflow_size"] = data["num_actions"] + data["num_decisions"] + data["num_forks"] + data["num_joins"]

    report["malformed_records"] = data.loc[data["malformed"], "origin"].tolist()[:200]
    return data, report


def describe_column(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {}
    return {"count": int(s.count()), "mean": float(s.mean()), "median": float(s.median()),
            "std": float(s.std()), "min": float(s.min()), "p25": float(s.quantile(.25)),
            "p75": float(s.quantile(.75)), "p95": float(s.quantile(.95)), "max": float(s.max())}


def dataset_statistics(data: pd.DataFrame) -> pd.DataFrame:
    groups = {name: g for name, g in data.groupby("dataset")}
    groups["ALL"] = data
    rows = []
    for name, g in groups.items():
        row = {
            "dataset": name,
            "workflows": len(g),
            "percent_of_total": 100 * len(g) / len(data),
            "unique_origins": g["origin"].nunique(),
            "missing_descriptions": int(g["missing_description"].sum()),
            "too_short_descriptions": int(g["too_short"].sum()),
            "malformed_records": int(g["malformed"].sum()),
            "unique_descriptions": g["description"].str.lower().nunique(),
            "duplicate_descriptions_beyond_first": int(g["duplicate_description"].sum()),
            "workflows_in_duplicate_groups": int(g["in_duplicate_group"].sum()),
            "used_in_clustering": int(g["used_in_clustering"].sum()),
            "has_original_prompt": int((g["original_prompt"].str.strip() != "").sum()),
            "pct_with_branching": 100 * g["has_branching"].mean(),
            "pct_with_parallelism": 100 * g["has_parallelism"].mean(),
            "pct_with_iteration": 100 * g["has_iteration"].mean(),
        }
        for col in ("n_words", "n_sentences", "num_actions", "num_decisions", "num_forks", "num_joins", "workflow_size"):
            for k, v in describe_column(g[col]).items():
                row[f"{col}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------------------
def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def corpus_fingerprint(texts: list[str], model: str, cfg: Config) -> str:
    h = hashlib.sha256()
    h.update(f"{model}|{cfg.long_text}|{cfg.chunk_overlap_tokens}".encode())
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()


def make_chunks(tokenizer, texts: list[str], window: int, overlap: int) -> tuple[list[str], list[int], np.ndarray]:
    """Split texts into overlapping token windows; returns chunk texts, owner index, token counts."""
    chunk_texts, owners, n_tokens = [], [], np.zeros(len(texts), dtype=np.int32)
    step = max(window - overlap, 1)
    for start in range(0, len(texts), 2000):
        batch = texts[start:start + 2000]
        enc = tokenizer(batch, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        for i, (ids, offsets) in enumerate(zip(enc["input_ids"], enc["offset_mapping"])):
            idx = start + i
            n_tokens[idx] = len(ids)
            if len(ids) <= window:
                chunk_texts.append(texts[idx]); owners.append(idx)
                continue
            for s in range(0, len(ids), step):
                e = min(s + window, len(ids))
                chunk_texts.append(texts[idx][offsets[s][0]:offsets[e - 1][1]]); owners.append(idx)
                if e == len(ids):
                    break
    return chunk_texts, owners, n_tokens


def select_sample(data: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    used = data[data["used_in_clustering"]]
    if cfg.sample_size and cfg.sample_size < len(used):
        return used.sample(n=cfg.sample_size, random_state=cfg.seed).sort_index()
    return used


def compute_embeddings(cfg: Config, paths: Paths) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    data = pd.read_parquet(paths.base)
    subset = select_sample(data, cfg)
    texts = subset["description"].tolist()
    model_name = resolve_model(cfg.model)
    fingerprint = corpus_fingerprint(texts, model_name, cfg)

    if cfg.reuse_cache and paths.emb_file.exists() and paths.emb_meta.exists():
        meta = json.loads(paths.emb_meta.read_text())
        if meta.get("fingerprint") == fingerprint:
            print(f"Reusing cached embeddings: {paths.emb_file}")
            return np.load(paths.emb_file)

    device = pick_device()
    model = SentenceTransformer(model_name, device=device)
    max_len = int(model.max_seq_length)
    window = max_len - 2
    print(f"model={model_name} device={device} max_seq_length={max_len} documents={len(texts)}")

    if cfg.long_text == "chunk":
        chunk_texts, owners, n_tokens = make_chunks(model.tokenizer, texts, window, cfg.chunk_overlap_tokens)
    else:
        enc = model.tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
        n_tokens = np.array([len(x) for x in enc], dtype=np.int32)
        chunk_texts, owners = texts, list(range(len(texts)))
    owners_arr = np.asarray(owners)
    truncated = int((n_tokens > window).sum())
    chunks_per_doc = np.bincount(owners_arr, minlength=len(texts))
    print(f"documents longer than {window} tokens: {truncated} ({100 * truncated / len(texts):.2f}%); "
          f"chunks to encode: {len(chunk_texts)}")

    part_dir = paths.embeddings / f"_parts_{paths.model_tag}_{fingerprint[:10]}"
    part_dir.mkdir(exist_ok=True)
    block = 4096
    parts = []
    for b, start in enumerate(range(0, len(chunk_texts), block)):
        part_file = part_dir / f"part_{b:05d}.npy"
        if part_file.exists():
            parts.append(np.load(part_file)); continue
        emb = model.encode(chunk_texts[start:start + block], batch_size=cfg.batch_size,
                           show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
        np.save(part_file, emb.astype(np.float32)); parts.append(emb.astype(np.float32))
        print(f"  encoded {min(start + block, len(chunk_texts))} / {len(chunk_texts)} chunks", flush=True)
    chunk_emb = np.concatenate(parts)

    # Token-length-weighted mean of chunk embeddings, re-normalised to unit length.
    lengths = np.array([max(len(t.split()), 1) for t in chunk_texts], dtype=np.float32)
    pooled = np.zeros((len(texts), chunk_emb.shape[1]), dtype=np.float32)
    np.add.at(pooled, owners_arr, chunk_emb * lengths[:, None])
    pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)

    np.save(paths.emb_file, pooled)
    subset[["origin"]].assign(row_position=subset.index).to_csv(paths.emb_index, index=False)
    paths.emb_meta.write_text(json.dumps({
        "model": model_name, "device": device, "max_seq_length": max_len, "window_tokens": window,
        "long_text_strategy": cfg.long_text, "chunk_overlap_tokens": cfg.chunk_overlap_tokens,
        "pooling": "length-weighted mean of chunk embeddings, L2-normalised",
        "normalized": True, "n_documents": len(texts), "n_chunks": len(chunk_texts),
        "documents_exceeding_window": truncated,
        "truncation_rate_if_not_chunked_percent": 100 * truncated / len(texts),
        "token_count": describe_column(pd.Series(n_tokens)),
        "fingerprint": fingerprint, "sample_size": cfg.sample_size, "seed": cfg.seed,
        "library_versions": library_versions(),
    }, indent=2))
    tok = pd.DataFrame({"origin": subset["origin"].values, "n_tokens": n_tokens, "n_chunks": chunks_per_doc})
    tok.to_csv(paths.embeddings / f"token_counts_{paths.model_tag}.csv", index=False)
    for f in part_dir.glob("*.npy"):
        f.unlink()
    part_dir.rmdir()
    return pooled


def library_versions() -> dict:
    import sklearn, torch, sentence_transformers, matplotlib
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "torch": torch.__version__,
            "sentence-transformers": sentence_transformers.__version__, "matplotlib": matplotlib.__version__}


def load_embeddings(cfg: Config, paths: Paths) -> tuple[np.ndarray, pd.DataFrame]:
    """Loads cached embeddings and verifies alignment with workflow identifiers."""
    data = pd.read_parquet(paths.base)
    emb = np.load(paths.emb_file)
    index = pd.read_csv(paths.emb_index)
    if len(index) != len(emb):
        raise ValueError("Embedding matrix and index length differ; rerun `embed`.")
    rows = data.loc[index["row_position"].values]
    if not (rows["origin"].values == index["origin"].values).all():
        raise ValueError("Embedding index no longer matches workflow identifiers; rerun `embed`.")
    return emb, rows


# ---------------------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------------------
def make_kmeans(cfg: Config, k: int, seed: int, n_init: int):
    from sklearn.cluster import KMeans, MiniBatchKMeans
    if cfg.algorithm == "minibatch":
        return MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=n_init, batch_size=4096)
    return KMeans(n_clusters=k, random_state=seed, n_init=n_init)


def size_balance(labels: np.ndarray) -> dict:
    counts = np.bincount(labels)
    p = counts / counts.sum()
    return {"smallest_cluster_pct": 100 * p.min(), "largest_cluster_pct": 100 * p.max(),
            "size_entropy_normalized": float(-(p * np.log(p)).sum() / np.log(len(p)))}


def sweep_k(cfg: Config, paths: Paths) -> pd.DataFrame:
    from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score

    emb, rows = load_embeddings(cfg, paths)
    rng = np.random.RandomState(cfg.seed)
    sil_idx = rng.choice(len(emb), size=min(cfg.silhouette_sample, len(emb)), replace=False)
    dataset_codes = pd.factorize(rows["dataset"])[0]
    results = []
    for k in range(cfg.k_min, cfg.k_max + 1):
        label_runs, sils, dbs, inertias = [], [], [], []
        for seed in cfg.sweep_seeds:
            km = make_kmeans(cfg, k, seed, n_init=1).fit(emb)
            labels = km.labels_
            label_runs.append(labels)
            sils.append(silhouette_score(emb[sil_idx], labels[sil_idx], metric="cosine"))
            dbs.append(davies_bouldin_score(emb, labels))
            inertias.append(km.inertia_)
        aris = [adjusted_rand_score(label_runs[i], label_runs[j])
                for i in range(len(label_runs)) for j in range(i + 1, len(label_runs))]
        row = {"k": k, "silhouette_mean": np.mean(sils), "silhouette_std": np.std(sils),
               "davies_bouldin_mean": np.mean(dbs), "inertia_mean": np.mean(inertias),
               "stability_ari_mean": np.mean(aris) if aris else np.nan,
               "nmi_with_dataset": normalized_mutual_info_score(dataset_codes, label_runs[0])}
        row.update(size_balance(label_runs[0]))
        results.append(row)
        print({k_: round(v, 4) if isinstance(v, float) else v for k_, v in row.items()}, flush=True)
    table = pd.DataFrame(results)
    # Rank-based suggestion only; interpretability decides the final K.
    ranks = (table["silhouette_mean"].rank(ascending=False) + table["davies_bouldin_mean"].rank()
             + table["stability_ari_mean"].rank(ascending=False))
    table["rank_score_lower_is_better"] = ranks
    table.to_csv(paths.tables / f"k_selection_{paths.model_tag}.csv", index=False)
    print(f"Metric-only suggestion: K={int(table.loc[ranks.idxmin(), 'k'])} (see k_selection table)")
    return table


def tokenize_for_keywords(stop_words: list[str]):
    from sklearn.feature_extraction.text import CountVectorizer
    return CountVectorizer(stop_words=stop_words, ngram_range=(1, 3), min_df=10, max_df=0.5,
                           max_features=150000, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b")


def cluster_keywords(texts: list[str], labels: np.ndarray, top_n: int = 25) -> pd.DataFrame:
    """Class-based TF-IDF (c-TF-IDF) keywords plus frequent bigrams/trigrams per cluster."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    import scipy.sparse as sp

    stop = sorted(set(ENGLISH_STOP_WORDS) | TEMPLATE_WORDS)
    vec = tokenize_for_keywords(stop)
    X = vec.fit_transform(texts)
    terms = np.asarray(vec.get_feature_names_out())
    k = labels.max() + 1
    ind = sp.csr_matrix((np.ones(len(labels)), (labels, np.arange(len(labels)))), shape=(k, len(labels)))
    class_tf = ind @ X                                           # term counts per cluster
    doc_freq = ind @ (X > 0).astype(np.float32)                  # documents containing the term
    class_tf_norm = sp.diags(1 / np.maximum(np.asarray(class_tf.sum(axis=1)).ravel(), 1)) @ class_tf
    avg_per_class = class_tf.sum() / k
    idf = np.log(1 + avg_per_class / np.maximum(np.asarray(class_tf.sum(axis=0)).ravel(), 1))
    ctfidf = class_tf_norm.multiply(idf).tocsr()
    sizes = np.bincount(labels, minlength=k)
    rows = []
    n_words = np.array([len(t.split(" ")) for t in terms])
    for c in range(k):
        scores = ctfidf[c].toarray().ravel()
        top = scores.argsort()[::-1][:top_n]
        df_c = np.asarray(doc_freq[c].todense()).ravel()
        bi = np.where(n_words == 2)[0]; tri = np.where(n_words == 3)[0]
        top_bi = bi[df_c[bi].argsort()[::-1][:10]]; top_tri = tri[df_c[tri].argsort()[::-1][:10]]
        rows.append({
            "cluster": c, "n_workflows": int(sizes[c]),
            "ctfidf_keywords": ", ".join(terms[top]),
            "frequent_bigrams": "; ".join(f"{terms[i]} ({df_c[i] / sizes[c]:.0%})" for i in top_bi),
            "frequent_trigrams": "; ".join(f"{terms[i]} ({df_c[i] / sizes[c]:.0%})" for i in top_tri),
        })
    return pd.DataFrame(rows)


def representative_examples(rows: pd.DataFrame, emb: np.ndarray, labels: np.ndarray, cfg: Config) -> pd.DataFrame:
    rng = np.random.RandomState(cfg.seed)
    out = []
    for c in range(labels.max() + 1):
        idx = np.where(labels == c)[0]
        centroid = emb[idx].mean(axis=0); centroid /= np.linalg.norm(centroid)
        sims = emb[idx] @ centroid
        order = idx[np.argsort(-sims)]
        picks = [("central", order[:cfg.n_representative])]
        spread = order[np.linspace(0, len(order) - 1, cfg.n_representative).astype(int)]
        picks.append(("spread_across_cluster", spread))
        picks.append(("random", rng.choice(idx, size=min(cfg.n_random_examples, len(idx)), replace=False)))
        for kind, sel in picks:
            for rank, i in enumerate(sel, 1):
                r = rows.iloc[i]
                out.append({"cluster": c, "kind": kind, "rank": rank, "origin": r["origin"],
                            "dataset": r["dataset"], "source": r["source"],
                            "similarity_to_centroid": float(emb[i] @ centroid),
                            "description": r["description"][:1500]})
    return pd.DataFrame(out)


def source_bias_report(rows: pd.DataFrame, labels: np.ndarray) -> tuple[pd.DataFrame, dict]:
    from scipy.stats import chi2_contingency, kruskal
    from sklearn.metrics import normalized_mutual_info_score

    ct = pd.crosstab(labels, rows["dataset"].values)
    chi2 = chi2_contingency(ct)[0]
    cramers_v = float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))))
    summary = {
        "nmi_cluster_vs_dataset": float(normalized_mutual_info_score(rows["dataset"], labels)),
        "nmi_cluster_vs_subsource": float(normalized_mutual_info_score(rows["source"], labels)),
        "cramers_v_cluster_vs_dataset": cramers_v,
        "kruskal_length_across_clusters_H": float(kruskal(*[rows["n_words"].values[labels == c] for c in np.unique(labels)])[0]),
        "share_starting_with_the_process_begins": float(rows["description"].str.startswith("The process begins").mean()),
    }
    share = ct.div(ct.sum(axis=1), axis=0)
    out = ct.copy(); out.columns = [f"n_{c}" for c in ct.columns]
    for c in ct.columns:
        out[f"share_of_cluster_{c}"] = share[c]
    out["dominant_dataset"] = share.idxmax(axis=1)
    out["dominant_share"] = share.max(axis=1)
    out["mean_words"] = [rows["n_words"].values[labels == c].mean() for c in out.index]
    out.index.name = "cluster"
    return out.reset_index(), summary


def run_cluster(cfg: Config, paths: Paths) -> None:
    emb, rows = load_embeddings(cfg, paths)
    km = make_kmeans(cfg, cfg.k, cfg.seed, cfg.n_init).fit(emb)
    labels = km.labels_
    np.save(paths.clusters, labels)

    kw = cluster_keywords(rows["description"].tolist(), labels)
    kw.to_csv(paths.tables / "cluster_keywords.csv", index=False)
    representative_examples(rows, emb, labels, cfg).to_csv(paths.tables / "cluster_representative_examples.csv", index=False)

    bias, bias_summary = source_bias_report(rows, labels)
    bias.to_csv(paths.tables / "cluster_source_distribution.csv", index=False)

    stats = rows.assign(cluster=labels).groupby("cluster").agg(
        n_workflows=("origin", "size"), mean_words=("n_words", "mean"), mean_actions=("num_actions", "mean"),
        mean_decisions=("num_decisions", "mean"), mean_forks=("num_forks", "mean"),
        pct_branching=("has_branching", "mean"), pct_parallel=("has_parallelism", "mean")).reset_index()
    stats["percent"] = 100 * stats["n_workflows"] / len(rows)
    stats.to_csv(paths.tables / "cluster_statistics.csv", index=False)

    # Near-duplicate / nearest-neighbour statistics via exact block-wise search (no N x N matrix).
    nn = 1 - nearest_neighbor_distances(emb)
    near = nn >= cfg.near_duplicate_threshold
    nn_table = rows.assign(nn_similarity=nn).groupby("dataset")["nn_similarity"].agg(["count", "mean", "median", "min", "max"])
    nn_table["near_duplicates"] = rows.assign(nd=near).groupby("dataset")["nd"].sum()
    nn_table["near_duplicate_pct"] = 100 * nn_table["near_duplicates"] / nn_table["count"]
    nn_table.to_csv(paths.tables / "nearest_neighbor_similarity_by_dataset.csv")

    tok_file = paths.embeddings / f"token_counts_{paths.model_tag}.csv"
    n_tokens = pd.read_csv(tok_file)["n_tokens"].values if tok_file.exists() else np.nan

    # Base per-workflow table: every loaded workflow is kept; unclustered ones get cluster_id = -1.
    data = pd.read_parquet(paths.base)
    cols = ["origin", "dataset", "source", "description", "original_prompt", "n_words", "n_sentences",
            "num_actions", "num_decisions", "num_forks", "num_joins", "has_branching", "has_parallelism",
            "has_iteration", "workflow_size", "missing_description", "too_short", "duplicate_description",
            "in_duplicate_group", "used_in_clustering"]
    out = data[cols].copy()
    out["cluster_id"] = -1
    out["n_tokens"] = np.nan
    out["nn_similarity"] = np.nan
    out.loc[rows.index, "cluster_id"] = labels
    out.loc[rows.index, "n_tokens"] = n_tokens
    out.loc[rows.index, "nn_similarity"] = nn
    out["category"] = ""
    out["category_status"] = ""
    out.to_csv(paths.assignments, index=False)

    summary = {"k": cfg.k, "inertia": float(km.inertia_), "bias": bias_summary,
               "near_duplicate_threshold": cfg.near_duplicate_threshold,
               "near_duplicates_total": int(near.sum()), "n_clustered": int(len(rows))}
    (paths.root / "clustering_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------------------
# Embedding-model stability comparison (shared reproducible sample)
# ---------------------------------------------------------------------------------------
def compare_models(cfg: Config) -> None:
    from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score

    res = {}
    labels = {}
    base_path = None
    for tag in ("minilm", "mpnet"):
        c = Config(**{**asdict(cfg), "model": tag, "sample_size": cfg.sample_size})
        p = Paths(c)
        base_path = p
        emb, rows = load_embeddings(c, p)
        km = make_kmeans(c, c.k, c.seed, c.n_init).fit(emb)
        labels[tag] = km.labels_
        sil_idx = np.random.RandomState(c.seed).choice(len(emb), size=min(c.silhouette_sample, len(emb)), replace=False)
        res[tag] = {"silhouette_cosine": float(silhouette_score(emb[sil_idx], km.labels_[sil_idx], metric="cosine")),
                    "davies_bouldin": float(davies_bouldin_score(emb, km.labels_)),
                    "origins": rows["origin"].tolist()}
    if res["minilm"]["origins"] != res["mpnet"]["origins"]:
        raise ValueError("Models were not embedded on the same sample (check --sample-size/--seed).")
    a, b = labels["minilm"], labels["mpnet"]
    ct = pd.crosstab(a, b)
    best_match = float((ct.max(axis=1).sum()) / ct.values.sum())
    out = {"sample_size": len(a), "k": cfg.k,
           "minilm": {k: v for k, v in res["minilm"].items() if k != "origins"},
           "mpnet": {k: v for k, v in res["mpnet"].items() if k != "origins"},
           "adjusted_rand_index": float(adjusted_rand_score(a, b)),
           "normalized_mutual_information": float(normalized_mutual_info_score(a, b)),
           "minilm_to_mpnet_best_match_purity": best_match}
    (base_path.root / "tables" / "embedding_model_comparison.json").write_text(json.dumps(out, indent=2))
    ct.to_csv(base_path.root / "tables" / "embedding_model_cluster_crosstab.csv")
    print(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------
def build_config(args) -> Config:
    cfg = Config()
    if args.config:
        cfg = Config(**{**asdict(cfg), **json.loads(Path(args.config).read_text())})
    for name in ("input_dir", "output_dir", "model", "batch_size", "long_text", "sample_size", "k_min",
                 "k_max", "k", "algorithm", "seed", "silhouette_sample", "n_init"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, name, value)
    if args.no_cache:
        cfg.reuse_cache = False
    if cfg.long_text not in {"chunk", "truncate"}:
        raise ValueError("long_text must be 'chunk' or 'truncate'")
    return cfg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["prepare", "embed", "sweep", "cluster", "compare"])
    p.add_argument("--config", help="JSON file overriding Config fields")
    p.add_argument("--input-dir", dest="input_dir")
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--model", help="minilm | mpnet | any sentence-transformers model name")
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--long-text", dest="long_text", choices=["chunk", "truncate"])
    p.add_argument("--sample-size", dest="sample_size", type=int)
    p.add_argument("--k-min", dest="k_min", type=int)
    p.add_argument("--k-max", dest="k_max", type=int)
    p.add_argument("--k", type=int)
    p.add_argument("--algorithm", choices=["kmeans", "minibatch"])
    p.add_argument("--n-init", dest="n_init", type=int)
    p.add_argument("--silhouette-sample", dest="silhouette_sample", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args(argv)

    cfg = build_config(args)
    if args.stage == "compare":
        compare_models(cfg)
        return 0
    paths = Paths(cfg)
    (paths.root / "experiment_configuration.json").write_text(json.dumps(
        {"config": asdict(cfg), "stage": args.stage, "library_versions": library_versions(),
         "model_resolved": resolve_model(cfg.model)}, indent=2, default=list))

    if args.stage == "prepare":
        data, report = load_enriched(cfg)
        data.to_parquet(paths.base)
        stats = dataset_statistics(data)
        stats.to_csv(paths.tables / "dataset_statistics.csv", index=False)
        dup = (data[data["in_duplicate_group"]].assign(key=lambda d: d["description"].str.lower())
               .groupby("key").agg(n=("origin", "size"), datasets=("dataset", lambda s: ",".join(sorted(set(s)))),
                                   example_origin=("origin", "first")).sort_values("n", ascending=False))
        dup.reset_index(drop=True).to_csv(paths.tables / "duplicate_statistics.csv", index=False)
        (paths.root / "data_loading_report.json").write_text(json.dumps(report, indent=2))
        print(stats.T.to_string())
        print(f"loaded={len(data)} used_in_clustering={int(data['used_in_clustering'].sum())} duplicate_groups={len(dup)}")
    elif args.stage == "embed":
        compute_embeddings(cfg, paths)
    elif args.stage == "sweep":
        sweep_k(cfg, paths)
    elif args.stage == "cluster":
        run_cluster(cfg, paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
