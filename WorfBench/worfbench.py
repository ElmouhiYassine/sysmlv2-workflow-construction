from collections import defaultdict, deque
from Transformers.graph_to_sysml import transform_into_sysml
from datasets import load_dataset
import csv
import re
import os
import statistics
import pandas as pd
from calculate_metrics import ProcessingMetrics, print_metrics


# Load train/test splits

def load_worfbench_splits():
    ds_train = load_dataset("zjunlp/WorFBench_train")

    if "test" in ds_train:
        return ds_train

    if "train" not in ds_train:
        raise RuntimeError("No 'train' split found in zjunlp/WorFBench_train")

    try:
        ds_test = load_dataset("zjunlp/WorFBench_test")
    except Exception:
        return ds_train

    train_split = ds_train["train"]
    test_split = ds_test["test"] if "test" in ds_test else ds_test.get("train")
    if test_split is None:
        return ds_train

    return {
        "train": train_split,
        "test": test_split,
    }


ds = load_worfbench_splits()


def get_conversation_list(example):
    #Return conversation list for either WorFBench schema.
    conversations = example.get("messages")
    if conversations is None:
        conversations = example.get("conversations")
    return conversations or []

def extract_user_prompt_and_source(example):
    conversations = get_conversation_list(example)
    source = example.get("source", "unknown")

    user_content = ""
    for msg in reversed(conversations):
        if msg.get("role") == "user":
            user_content = str(msg.get("content", "")).strip()
            break

    if "Now it's your turn.\n" in user_content:
        tail = user_content.split("Now it's your turn.\n", 1)[1]
        prompt = ",".join(tail.split(",")[:-1]).strip()
    else:
        m = re.search(r"Your task is to:\s*(.+?)(?:\.?\s*The action list|$)", user_content, flags=re.DOTALL)
        prompt = m.group(1).strip() if m else user_content

    if prompt.startswith("Task:"):
        prompt = prompt[5:].strip()

    return prompt, source



# function to extract the nodes (actions) and the edges from one dataset example
def extract_nodes_and_edges(example):
    conversations = get_conversation_list(example)

    a_m = None
    for msg in reversed(conversations):
        if msg["role"] == "assistant":
            a_m = msg["content"]
            break
    if a_m is None:
        return None

    text = a_m.strip()

    # Nodes
    m = re.search(r'(?im)^\s*Node\s*:\s*', text)
    node_section = text
    if m:
        tail = text[m.end():]
        stop = re.search(r'(?im)^\s*(?:\*\*.*Edge.*\*\*|Edge|Edges|###)\b', tail)
        node_section = tail[:stop.start()] if stop else tail

    node_pattern_colon = re.compile(r'^\s*(\d+)\s*:\s*(.+)$', re.MULTILINE)
    found = node_pattern_colon.findall(node_section)

    if not found:
        node_pattern_any = re.compile(r'^\s*(\d+)\s*[\.:]\s*(.+)$', re.MULTILINE)
        found = node_pattern_any.findall(node_section)

    nodes = {str(int(i)): content.strip() for i, content in found}

    edge_tuple_re = re.compile(r"\(\s*((?:START|END|\d+))\s*,\s*((?:START|END|\d+))\s*\)")

    edges_header = re.search(
        r'(?im)^\s*(?:\*\*)?\s*Edges?\s*(?:\*\*)?\s*:\s*',
        text
    )

    if edges_header:
        edge_part = text[edges_header.end():]
    else:
        edge_marker = re.search(r'(?im)^\s*Edge\s*:\s*', text)
        if edge_marker:
            edge_part = text[edge_marker.end():]
        else:
            edge_lines = "\n".join(re.findall(r'(?im)^\s*[-*]?\s*.*\bEdge\b.*$', text))
            edge_part = edge_lines

    if edge_part:
        stop = re.search(r'(?im)^\s*(Node|Nodes|Graph|Notes?|Output|Example)\s*:\s*', edge_part)
        if stop:
            edge_part = edge_part[:stop.start()]

    edges = [(a.strip(), b.strip()) for a, b in edge_tuple_re.findall(edge_part)]

    indeg_p = defaultdict(int)
    indeg_c = defaultdict(int)
    nodes_in_edges = set()

    for u, v in edges:
        indeg_c[u] += 1
        indeg_p[v] += 1
        nodes_in_edges.add(u)
        nodes_in_edges.add(v)

    start_nodes = [n for n in nodes_in_edges if indeg_p[n] == 0 and n != "START"]
    end_nodes = [n for n in nodes_in_edges if indeg_c[n] == 0 and n != "END"]

    # add START/END only if not present already
    if not any(u == "START" for u, _ in edges):
        for n in start_nodes:
            edges.append(("START", n))

    if not any(v == "END" for _, v in edges):
        for n in end_nodes:
            edges.append((n, "END"))

    return {"nodes": nodes, "edges": edges}



# function used only to print nodes and edges in a readable way
def format_graph(graph):
    out = []

    out.append("Nodes:")
    for i in graph["nodes"].keys():
        out.append(f"{i}. {graph['nodes'][i]}")

    out.append("\nEdges:")
    for e in graph["edges"]:
        out.append(f"{e}")


    return "\n".join(out)

words = [" if ", " repeat ", " else ", " while ", " until ", " for each "]
# function to count the occurrences of each keyword in 'words' across the dataset
def count_words():
    # count = defaultdict(int)

    # for example in ds[split]:
    #     graph = extract_nodes_and_edges(example)
    #     if sources_dic[example["source"]] is None:
    #         sources_dic[example["source"]] = graph

    for split_name in ["train", "test"]:
        if split_name not in ds:
            continue
        for i in range(len(ds[split_name])):
            n = extract_nodes_and_edges(ds[split_name][i])

            for j in n["nodes"].values():

                if " repeat " in j:
                    print(format_graph(n))
                    break

sources = set()
sources_dic = {}
for i in ['toolalpaca', 'environment/os', 'wikihow', 'environment/webshop', 'toolbench', 'environment/alfworld', 'lumos']:
    sources_dic[i] = None

def edge_nodes(edges):
    nodes = set()
    for a, b in edges:
        nodes.add(a)
        nodes.add(b)
    return nodes


def summarize_counts(values):
    if not values:
        return {
            "count": 0,
            "sum": 0,
            "mean": 0.0,
            "min": 0,
            "max": 0,
            "variance": 0.0,
        }

    return {
        "count": len(values),
        "sum": sum(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "variance": statistics.pvariance(values) if len(values) > 1 else 0.0,
    }


def count_flow_elements(graph):
    nodes = graph["nodes"]
    edges = graph["edges"]

    action_nodes = [n for n in nodes.keys() if n not in {"START", "END"}]
    actions = len(action_nodes)

    indeg = defaultdict(int)
    outdeg = defaultdict(int)
    for src, dst in edges:
        outdeg[src] += 1
        indeg[dst] += 1

    forks = sum(1 for n in action_nodes if outdeg[n] > 1)
    joins = sum(1 for n in action_nodes if indeg[n] > 1)
    # WorFBench plans do not encode explicit decision gateway nodes.
    decisions = 0

    return actions, decisions, joins, forks


def print_flow_metrics(action_counts, decision_counts, join_counts, fork_counts):
    valid_models = len(action_counts)
    action_stats = summarize_counts(action_counts)
    decision_stats = summarize_counts(decision_counts)
    join_stats = summarize_counts(join_counts)
    fork_stats = summarize_counts(fork_counts)

    total_actions = action_stats["sum"]
    total_decisions = decision_stats["sum"]
    total_joins = join_stats["sum"]
    total_forks = fork_stats["sum"]
    all_flow_items = total_actions + total_decisions + total_joins + total_forks

    by_model = {
        "actions": (sum(1 for x in action_counts if x > 0) / valid_models) if valid_models else 0.0,
        "decisions": (sum(1 for x in decision_counts if x > 0) / valid_models) if valid_models else 0.0,
        "joins": (sum(1 for x in join_counts if x > 0) / valid_models) if valid_models else 0.0,
        "forks": (sum(1 for x in fork_counts if x > 0) / valid_models) if valid_models else 0.0,
    }

    global_freq = {
        "actions": (total_actions / all_flow_items) if all_flow_items else 0.0,
        "decisions": (total_decisions / all_flow_items) if all_flow_items else 0.0,
        "joins": (total_joins / all_flow_items) if all_flow_items else 0.0,
        "forks": (total_forks / all_flow_items) if all_flow_items else 0.0,
    }

    print(
        f"actions: sum={action_stats['sum']} mean={action_stats['mean']:.3f} "
        f"min={action_stats['min']} max={action_stats['max']} var={action_stats['variance']:.3f}"
    )
    print(
        f"decisions: sum={decision_stats['sum']} mean={decision_stats['mean']:.3f} "
        f"min={decision_stats['min']} max={decision_stats['max']} var={decision_stats['variance']:.3f}"
    )
    print(
        f"joins: sum={join_stats['sum']} mean={join_stats['mean']:.3f} "
        f"min={join_stats['min']} max={join_stats['max']} var={join_stats['variance']:.3f}"
    )
    print(
        f"forks: sum={fork_stats['sum']} mean={fork_stats['mean']:.3f} "
        f"min={fork_stats['min']} max={fork_stats['max']} var={fork_stats['variance']:.3f}"
    )
    print(f"frequencies by model: {by_model}")
    print(f"frequencies global: {global_freq}")


BAD_FILE = "bad_instances.csv"


def log_bad(instance_id, reason):
    with open(BAD_FILE, mode="a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["instance_id", "reason"])
        if f.tell() == 0:
            w.writeheader()
        w.writerow({"instance_id": instance_id, "reason": reason})

if __name__ == "__main__":
    drop_sources = {"environment/alfworld"}
    filename = "worfbench_sysml.csv"
    headers = ["data", "source", "user_prompt", "sysml_code"]

    splits_to_process = [s for s in ["train", "test"] if s in ds]
    total_instances = sum(len(ds[s].filter(lambda x: x["source"] not in drop_sources)) for s in splits_to_process)
    metrics = ProcessingMetrics(total_instances=total_instances)

    action_counts = []
    decision_counts = []
    join_counts = []
    fork_counts = []

    print(f"Found {total_instances} instances across splits: {', '.join(splits_to_process)}")

    for split_name in splits_to_process:
        ds_filtered = ds[split_name].filter(lambda x: x["source"] not in drop_sources)
        len_ds = len(ds_filtered)
        print(f"Processing split '{split_name}' with {len_ds} instances")

        for idx in range(len_ds):
            if metrics.total_processed % 1000 == 0 and metrics.total_processed > 0:
                print(f"processed {metrics.total_processed} | OK: {metrics.ok} | Nodes Issue: {metrics.nodes_issue}")

            metrics.total_processed += 1

            instance = ds_filtered[idx]

            user_prompt, source = extract_user_prompt_and_source(instance)

            graph = extract_nodes_and_edges(instance)
            if graph is None:
                metrics.graph_none += 1
                log_bad(f"{split_name}:{idx}", "graph is None")
                continue

            declared = set(graph["nodes"].keys()) | {"START", "END"}
            referenced = {x for e in graph["edges"] for x in e}

            if referenced != declared:
                metrics.nodes_issue += 1
                log_bad(f"{split_name}:{idx}", f"node mismatch: declared={len(declared)} referenced={len(referenced)}")
                continue

            try:
                sysml_code = transform_into_sysml(graph)
            except Exception as e:
                metrics.transform_fail += 1
                log_bad(f"{split_name}:{idx}", f"{type(e).__name__}: {str(e)}")
                continue

            metrics.ok += 1

            actions, decisions, joins, forks = count_flow_elements(graph)
            action_counts.append(actions)
            decision_counts.append(decisions)
            join_counts.append(joins)
            fork_counts.append(forks)

            row = [
                {
                    "data": "WorfBench",
                    "source": source,
                    "user_prompt": user_prompt,
                    "sysml_code": sysml_code
                }
            ]
            file_exists = os.path.exists(filename)

            with open(filename, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=headers)

                if not file_exists:
                    writer.writeheader()

                writer.writerows(row)

    # Print final metrics summary
    print("\nmetrics:")
    print_metrics(metrics)

    print("element metrics:")
    print_flow_metrics(action_counts, decision_counts, join_counts, fork_counts)
    
    df_check = pd.read_csv(filename)
    print(f"\nTotal in output CSV: {len(df_check)}")
