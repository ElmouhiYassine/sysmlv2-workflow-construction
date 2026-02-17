from collections import defaultdict, deque
from Transformers.graph_to_sysml import transform_into_sysml
from datasets import load_dataset
import re
import json
import csv

import subprocess
import sys
ds = load_dataset("zjunlp/WorFBench_train")

def extract_user_prompt_and_source(example):
    conversations = example["messages"]
    source = example["source"]
    reversed(conversations)
    msg = conversations[1]['content'].split('\n')
    prompt = msg[1].split(':')[1]
    return prompt, source



# function to extract the nodes (actions) and the edges from one dataset example
import re

import re

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
        # stop at Edge / ### / etc. (keep your stop if you want)
        stop = re.search(r'(?im)^\s*(?:\*\*.*Edge.*\*\*|Edge|Edges|###)\b', tail)
        node_section = tail[:stop.start()] if stop else tail

    # 1) strict colon nodes
    node_pattern_colon = re.compile(r'^\s*(\d+)\s*:\s*(.+)$', re.MULTILINE)
    found = node_pattern_colon.findall(node_section)

    # 2) fallback dot/colon nodes only if colon style not found
    if not found:
        node_pattern_any = re.compile(r'^\s*(\d+)\s*[\.:]\s*(.+)$', re.MULTILINE)
        found = node_pattern_any.findall(node_section)

    nodes = {str(int(i)): content.strip() for i, content in found}

    # Edges (strict)
    edge_tuple_re = re.compile(r"\(\s*((?:START|END|\d+))\s*,\s*((?:START|END|\d+))\s*\)")

    # 1) Prefer "Edges" section
    edges_header = re.search(
        r'(?im)^\s*(?:\*\*\s*Edges\s*:\s*\*\*|\*\*\s*Edges\s*\*\*\s*:|Edges\s*:)\s*',
        text
    )

    if edges_header:
        edge_part = text[edges_header.end():]
    else:
        # 2) Fallback: after "Edge:"
        edge_marker = re.search(r'(?im)^\s*Edge\s*:\s*', text)
        if edge_marker:
            edge_part = text[edge_marker.end():]
        else:
            # 3) Fallback: lines that mention "Edge" (e.g., "- **Edge**: (1,2)")
            edge_lines = "\n".join(re.findall(r'(?im)^\s*[-*]?\s*.*\bEdge\b.*$', text))
            edge_part = edge_lines

    # stop at next header (optional)
    if edge_part:
        stop = re.search(r'(?im)^\s*(Node|Nodes|Graph|Notes?|Output|Example)\s*:\s*', edge_part)
        if stop:
            edge_part = edge_part[:stop.start()]

    edges = [(a.strip(), b.strip()) for a, b in edge_tuple_re.findall(edge_part)]

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

    # with open("test1.txt", "w", encoding="utf-8") as f:
    #     f.write("".join("\n".join(out)))

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

BAD_FILE = "bad_instance.csv"

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

    # Choix instance
    # 191
    # 225
    # 22
    # 505
    # idx = 520

    # c_graph = {'nodes': {'1': 'action1.','2': 'action2.', '3': 'action3.','4': 'action3.', '5': 'action5.', '6': 'action6.','7': 'action7.', '8': 'action8.'},
    #             'edges': [('START', '1'), ('1', '2'), ('1', '5'), ('2', '3'), ('2', '4'), ('3', '8'), ('5', '4'),('4', '7'), ('5', '6'), ('6', '7'), ('7', '8'), ('8', 'END')]}
    #

    filename = "worfbench_sysml.csv"
    headers = ["data", "source", "user_prompt", "sysml_code"]

    print(len(ds_filtered))
    a = ds_filtered[9774]['messages'][-1]
    print(a)
    print(format_graph(extract_nodes_and_edges(ds_filtered[9774])))
    graph_none = 0
    nodes_issue = 0
    for idx in range(len(ds_filtered)):

        instance = ds_filtered[idx]
        user_prompt, source = extract_user_prompt_and_source(instance)
        graph = extract_nodes_and_edges(instance)

        if graph is None:
            graph_none += 1
            log_bad(idx, "graph is None")
            print('graph issue')
            continue

        declared = set(graph["nodes"].keys()) | {"START", "END"}
        referenced = {x for e in graph["edges"] for x in e}


        if referenced != declared:
            nodes_issue += 1
            log_bad(idx, f"node mismatch: declared={len(declared)} referenced={len(referenced)}")
            print('nodes issue')
            continue
        try:
            sysml_code = transform_into_sysml(graph)
        except Exception as e:
            log_bad(idx, f"{type(e).__name__}: {e}")
            print(f" transform failed for {idx}: {e}")
            continue
        row = [
            {
                "data": "WorfBench",
                "source": source,
                "user_prompt": user_prompt,
                "sysml_code": sysml_code
            }
        ]
        with open(filename, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            if file.tell() == 0:
                writer.writeheader()
            writer.writerows(row)



