from collections import defaultdict, deque
from Transformers.graph_to_sysml import transform_into_sysml
from datasets import load_dataset
import csv
import re
import os
import pandas as pd
import subprocess
import sys
ds = load_dataset("zjunlp/WorFBench_train")

def extract_user_prompt_and_source(example):
    conversations = example["messages"]
    source = example["source"]
    msg = conversations[-2]['content'].split('Now it\'s your turn.\n')
    prompt = ",".join(msg[1].split(',')[0:-1])
    if prompt[0:5] == 'Task:':
        prompt = prompt[5:]
    return prompt, source



# function to extract the nodes (actions) and the edges from one dataset example
def extract_nodes_and_edges(example):
    conversations = example["messages"]

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
        out.append(f"{i}. {graph["nodes"][i]}")

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

    for i in range(len(ds['train'])):
        n = extract_nodes_and_edges(ds['train'][i])

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

BAD_FILE = "bad_instances.csv"

def log_bad(instance_id, reason):
    with open(BAD_FILE, mode="a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["instance_id", "reason"])
        if f.tell() == 0:
            w.writeheader()
        w.writerow({"instance_id": instance_id, "reason": reason})
if __name__ == "__main__":
    split = list(ds.keys())[0]   # "train"

    drop_sources = {"environment/alfworld"}
    ds_filtered = ds[split].filter(lambda x: x["source"] not in drop_sources)


    filename = "worfbench_sysmls.csv"
    headers = ["data", "source", "user_prompt", "sysml_code"]

    len_ds = len(ds_filtered)
    print(f"Found {len_ds} instances.")

    ok = 0
    graph_none = 0
    nodes_issue = 0
    transform_fail = 0

    k = 0

    for idx in range(len_ds):
        if (idx + 1) % 1000 == 0:
            print("processed", idx + 1, "OK", ok, "nodes_issue", nodes_issue)

        instance = ds_filtered[idx]

        user_prompt, source = extract_user_prompt_and_source(instance)

        graph = extract_nodes_and_edges(instance)
        if graph is None:
            graph_none += 1
            log_bad(idx, "graph is None")
            continue

        declared = set(graph["nodes"].keys()) | {"START", "END"}
        referenced = {x for e in graph["edges"] for x in e}

        if referenced != declared:
            nodes_issue += 1
            log_bad(idx, f"node mismatch: declared={len(declared)} referenced={len(referenced)}")
            continue

        try:
            sysml_code = transform_into_sysml(graph)
        except Exception:
            transform_fail += 1
            log_bad(idx, f"{type(Exception).__name__}: {Exception}")
            continue

        ok += 1
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

    print("Total:", len_ds)
    print("OK:", ok)
    print("graph_none:", graph_none)
    print("nodes_issue:", nodes_issue)
    print("transform_fail:", transform_fail)
    print("Sum:", ok + graph_none + nodes_issue + transform_fail)

    df_check = pd.read_csv(filename)
    print(len(df_check))
