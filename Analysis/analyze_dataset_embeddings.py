from pathlib import Path
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

INPUT_FILES = [
    "sopbench_sysmlv2.csv",
    "worfbench_sysmlv2.csv",
    "sapsam_sysmlv2.csv",
]

OUTPUT_DIR = Path("dataset_analysis")


def get_dataset_from_origin(origin: str) -> str:
    if origin.startswith("sopbench"):
        return "sopbench"
    if origin.startswith("worfbench"):
        return "worfbench"
    if origin.startswith("sapsam"):
        return "sapsam"
    return "unknown"


def get_source_from_origin(origin: str) -> str:
    parts = origin.split("::")

    if parts[0] == "worfbench" and len(parts) >= 2:
        return parts[1]

    if parts[0] == "sopbench":
        return "sopbench"

    if parts[0] == "sapsam":
        return "sapsam"

    return "unknown"


def count_actions(sysml):
    pattern = r"\baction\s+[A-Za-z_][A-Za-z0-9_]*\s*\{"
    return len(re.findall(pattern, sysml))


def count_decisions(sysml):
    pattern = r"\bdecide\s+[A-Za-z_][A-Za-z0-9_]*"
    return len(re.findall(pattern, sysml))


def count_forks(sysml):
    pattern = r"\bthen\s+fork\s*;"
    return len(re.findall(pattern, sysml))


def count_joins(sysml):
    pattern = r"\bjoin\s+[A-Za-z_][A-Za-z0-9_]*\s*;"
    return len(re.findall(pattern, sysml))

def load_all_data(paths):
    dfs = []

    for path in paths:
        path = Path(path)
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        df = pd.read_csv(path)

        required = ["origin", "user_prompt", "sysml_model"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"{path} missing column: {col}")

        dfs.append(df[required])

    if not dfs:
        raise ValueError("No dataset files found.")

    data = pd.concat(dfs, ignore_index=True)

    data["user_prompt"] = data["user_prompt"].fillna("").astype(str)
    data["sysml_model"] = data["sysml_model"].fillna("").astype(str)
    data["dataset"] = data["origin"].apply(get_dataset_from_origin)
    data["source"] = data["origin"].apply(get_source_from_origin)

    return data


def add_structure_features(df):
    df["prompt_length"] = df["user_prompt"].str.split().apply(len)
    df["sysml_length"] = df["sysml_model"].str.split().apply(len)

    df["num_actions"] = df["sysml_model"].apply(count_actions)
    df["num_decisions"] = df["sysml_model"].apply(count_decisions)
    df["num_forks"] = df["sysml_model"].apply(count_forks)
    df["num_joins"] = df["sysml_model"].apply(count_joins)

    return df


def compute_embeddings(df, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)

    df["embedding_text"] = df.apply(build_embedding_text, axis=1)
    prompts = df["embedding_text"].tolist()

    embeddings = model.encode(
        prompts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return embeddings




def build_embedding_text(row):
    user_prompt = str(row.get("user_prompt", "")).strip()

    if user_prompt:
        return user_prompt

    sysml = str(row.get("sysml_model", ""))

    descriptions = re.findall(
        r'description\s*=\s*"([^"]*)"',
        sysml,
        flags=re.DOTALL,
    )

    actions = re.findall(
        r"\baction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{",
        sysml,
    )

    decisions = re.findall(
        r'\bif\s+condition\s+"([^"]*)"',
        sysml,
    )

    parts = []

    if descriptions:
        parts.extend(descriptions)

    if actions:
        parts.extend(actions)

    if decisions:
        parts.extend(decisions)

    return " ".join(parts).strip()

def reduce_with_pca(embeddings):
    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(embeddings)

    return points, pca.explained_variance_ratio_


def cluster_embeddings(embeddings, n_clusters=8):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)

    score = silhouette_score(embeddings, labels)

    return labels, score


def plot_by_column(df, x_col, y_col, color_col, output_path):
    plt.figure(figsize=(10, 8))

    for value in sorted(df[color_col].unique()):
        subset = df[df[color_col] == value]
        plt.scatter(
            subset[x_col],
            subset[y_col],
            label=value,
            alpha=0.6,
            s=8,
        )

    plt.legend(markerscale=2, fontsize=8)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"Prompt embedding space colored by {color_col}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def nearest_neighbor_distances(embeddings, block_size=1024):
    """Exact nearest-neighbour cosine distance computed block-wise.

    Requires L2-normalised embeddings. Memory stays at block_size x N instead of N x N.
    """
    n = len(embeddings)
    nearest_sim = np.empty(n, dtype=np.float32)

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        block = embeddings[start:stop] @ embeddings.T
        block[np.arange(stop - start), np.arange(start, stop)] = -1
        nearest_sim[start:stop] = block.max(axis=1)

    return 1 - nearest_sim

def save_cluster_source_distribution(df):
    table = pd.crosstab(df["cluster"], df["source"])
    table.to_csv(OUTPUT_DIR / "cluster_source_distribution.csv")


def save_top_keywords_by_cluster(df, top_n=20):
    rows = []

    for cluster_id in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == cluster_id]
        texts = cluster_df["embedding_text"].fillna("").astype(str).tolist()

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=top_n,
            ngram_range=(1, 2),
        )

        try:
            X = vectorizer.fit_transform(texts)
            scores = np.asarray(X.mean(axis=0)).ravel()
            terms = vectorizer.get_feature_names_out()

            top_indices = scores.argsort()[::-1][:top_n]
            keywords = [terms[i] for i in top_indices]
        except ValueError:
            keywords = []

        rows.append({
            "cluster": cluster_id,
            "top_keywords": ", ".join(keywords),
        })

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "cluster_top_keywords.csv",
        index=False,
    )


def save_representative_examples_by_cluster(df, embeddings, top_n=10):
    rows = []

    for cluster_id in sorted(df["cluster"].unique()):
        indices = df.index[df["cluster"] == cluster_id].to_numpy()
        cluster_embeddings = embeddings[indices]

        centroid = cluster_embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        similarities = cluster_embeddings @ centroid
        best_local_indices = similarities.argsort()[::-1][:top_n]

        for rank, local_idx in enumerate(best_local_indices, start=1):
            global_idx = indices[local_idx]
            row = df.loc[global_idx]

            rows.append({
                "cluster": cluster_id,
                "rank": rank,
                "origin": row["origin"],
                "source": row["source"],
                "dataset": row["dataset"],
                "embedding_text": row["embedding_text"],
                "user_prompt": row["user_prompt"],
                "num_actions": row["num_actions"],
                "num_decisions": row["num_decisions"],
                "num_forks": row["num_forks"],
            })

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "cluster_representative_examples.csv",
        index=False,
    )
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all_data(INPUT_FILES)
    df = add_structure_features(df)

    print("Loaded rows:", len(df))
    print("\nRows by dataset:")
    print(df["dataset"].value_counts())

    print("\nRows by source:")
    print(df["source"].value_counts())

    print("\nStructure statistics by source:")
    stats = df.groupby("source")[
        [
            "prompt_length",
            "sysml_length",
            "num_actions",
            "num_decisions",
            "num_forks",
            "num_joins",
        ]
    ].agg(["count", "mean", "min", "max"])

    print(stats)

    stats.to_csv(OUTPUT_DIR / "structure_statistics_by_source.csv")

    print("\nComputing embeddings...")
    embeddings = compute_embeddings(df)

    np.save(OUTPUT_DIR / "prompt_embeddings.npy", embeddings)

    points, explained = reduce_with_pca(embeddings)
    df["pca_x"] = points[:, 0]
    df["pca_y"] = points[:, 1]

    print("PCA explained variance:", explained)

    labels, sil_score = cluster_embeddings(embeddings, n_clusters=8)
    df["cluster"] = labels

    save_cluster_source_distribution(df)
    save_top_keywords_by_cluster(df, top_n=20)
    save_representative_examples_by_cluster(df, embeddings, top_n=10)

    print("Silhouette score:", sil_score)

    nn_dist = nearest_neighbor_distances(embeddings)
    df["nearest_neighbor_distance"] = nn_dist

    print("\nNearest-neighbor distance by source:")
    nn_stats = df.groupby("source")["nearest_neighbor_distance"].agg(
        ["count", "mean", "min", "max"]
    )
    print(nn_stats)

    nn_stats.to_csv(OUTPUT_DIR / "nearest_neighbor_distance_by_source.csv")

    df.to_csv(OUTPUT_DIR / "dataset_with_embedding_analysis.csv", index=False)

    plot_by_column(
        df,
        "pca_x",
        "pca_y",
        "dataset",
        OUTPUT_DIR / "embedding_by_dataset.png",
    )

    plot_by_column(
        df,
        "pca_x",
        "pca_y",
        "source",
        OUTPUT_DIR / "embedding_by_source.png",
    )

    plot_by_column(
        df,
        "pca_x",
        "pca_y",
        "cluster",
        OUTPUT_DIR / "embedding_by_cluster.png",
    )

    cluster_summary = df.groupby("cluster").agg(
        count=("cluster", "count"),
        main_source=("source", lambda x: x.value_counts().index[0]),
        avg_actions=("num_actions", "mean"),
        avg_decisions=("num_decisions", "mean"),
        avg_forks=("num_forks", "mean"),
        avg_prompt_length=("prompt_length", "mean"),
    )

    cluster_summary.to_csv(OUTPUT_DIR / "cluster_summary.csv")

    print("\nSaved outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()