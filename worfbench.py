from collections import defaultdict, deque

from datasets import load_dataset
import re
import json

import subprocess
import sys
ds = load_dataset("zjunlp/WorFBench_train")

# function to extract the nodes (actions) and the edges from one dataset example
def extract_nodes_and_edges(example):
    conversations = example["messages"]

    # nodes and edges are in the last assistant message of the conversation
    a_m = None
    for msg in reversed(conversations):
        if msg["role"] == "assistant":
            a_m = msg["content"]
            break

    if a_m is None:
        return None

    text = a_m.strip()

    # nodes
    node_pattern = re.compile(r'^\s*(\d+)[\.:]\s*(.+)$', re.MULTILINE)
    nodes = [(int(i), content.strip()) for i, content in node_pattern.findall(text)]

    #edge
    edge_line_match = re.search(r'(?mi)^\s*Edge:\s*(.*)$', text)
    edge_part = edge_line_match.group(1) if edge_line_match else ""

    edge_pattern = re.compile(r"\(\s*([^,\)]+?)\s*,\s*([^)]+?)\s*\)")
    edges = [(a.strip(), b.strip()) for a, b in edge_pattern.findall(edge_part)]

    nodes = {str(i): txt for i, txt in nodes}

    return {
        "nodes": nodes,
        "edges": edges
    }

# function used only to print nodes and edges in a readable way
def format_graph(graph):
    out = []

    out.append("Nodes:")
    for i in graph["nodes"].keys():
        out.append(f"{i}. {graph["nodes"][i]}")

    out.append("\nEdges:")
    for e in graph["edges"]:
        out.append(f"{e}")

    with open("worfbench", "w", encoding="utf-8") as f:
        f.write("".join("\n".join(out)))

    return "\n".join(out)


words = [" if ", " repeat ", " else ", " while ", " until ", " for each "]
# function to count the occurrences of each keyword in 'words' across the dataset
def count_words(words):
    count = defaultdict(int)

    for example in ds[split]:
        graph = extract_nodes_and_edges(example)
        if sources_dic[example["source"]] is None:
            sources_dic[example["source"]] = graph

        for i in graph["nodes"]:
            for w in words:
                if w in i[1].lower():
                    count[w] = count[w] + 1

    return count, sources

sources = set()
sources_dic = {}
for i in ['toolalpaca', 'environment/os', 'wikihow', 'environment/webshop', 'toolbench', 'environment/alfworld', 'lumos']:
    sources_dic[i] = None




nodes = {}

def transfrom_into_sysml(graph: dict, package_name="PackageName", process_name="ProcessName"):

    def indent(level):
        return "    " * level

    node_text = graph["nodes"]
    edges = [(u.strip(), v.strip()) for (u, v) in graph["edges"]]

    # function to escape backslashes and double quotes
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    # function that computes successors and predecessors for each node
    def succ_and_pred():
        succ = defaultdict(list)
        pred = defaultdict(list)
        for u, v in edges:
            succ[u].append(v)
            pred[v].append(u)

        return succ, pred

    # function that generates the SysML action definitions
    def action_def():
        action_defs = []
        for node in sorted(node_text.keys(), key=sort_key):
            if len(pred[node]) > 1 or len(succ[pred[node][0]]) == 1:
                prefix = f"{indent(2)}then "
            else :
                prefix = f"{indent(2)}"
            d = (
                f'{prefix}action action{node} {{\n'
                f'{indent(3)}description = "{esc(node_text[node])}";\n'
                f'{indent(2)}}}\n'
            )
            action_defs.append(d)
        return action_defs

    def sort_key(x):
        return int(x) if str(x).isdigit() else str(x)

    succ, pred = succ_and_pred()
    action_defs = action_def()

    # Function that checks whether an action is a fork node and returns the fork line if true
    def fork(node):
        if len(succ[node]) > 1:
            line = f'{indent(2)}then fork;'
            for suc in succ[node]:
                line += f' then action{suc};'
            line += "\n"
            return True,line
        else:
            return False, ""

    exec_lines = []  # of generated SysML execution lines
    arrivals = defaultdict(int) # arrival counter: how many incoming branches reached each join node
    visited = set() # set of nodes already emitted

    # emits the SysML code (definition) for one action node, and marks it as visited
    def emit_action(n):
        if n == 'END' or n in visited:
            return
        exec_lines.append(action_defs[int(n) - 1])
        visited.add(n)

    # DFS traversal:
    # - on join nodes: emit "then joinNodeX" for each incoming branch, and emit "join joinNodeX" only when all
    #   predecessors have arrived, then continue from that node
    # - on fork nodes: emit fork line and recursively traverse each outgoing branch
    # - on linear nodes: continue to the single successor
    def dfs(node):
        if len(pred[node]) > 1:
            exec_lines.append(f"{indent(2)}then joinNode{node};\n")
            arrivals[node] += 1

            if arrivals[node] == len(pred[node]):
                exec_lines.append(f"{indent(2)}join  joinNode{node};\n")

                emit_action(node)

                for s in succ.get(node, []):
                    dfs(s)
            return
        emit_action(node)

        f, fl = fork(node)

        if f:
            exec_lines.append(fl)
            for s in succ[node]:
                dfs(s)
            return

        if len(succ[node]) == 1:
            dfs(succ[node][0])

    # dfs call
    f_start, fl_start = fork('START')
    if f_start:
        exec_lines.append(fl_start)
        for suc in succ['START']:
            dfs(suc)
    else:
        dfs(succ['START'][0])

    # sysml code construction
    lines = []
    lines.append(f"package {package_name} {{\n\n")
    lines.append(f"{indent(1)}action def {process_name} {{\n\n")

    lines.append(f"\n{indent(2)}first start;\n\n")

    lines.extend(exec_lines)

    lines.append(f"\n{indent(2)}then done;\n")
    lines.append(f"{indent(1)}}}\n")
    lines.append("}\n")

    with open("sysml.sysml", "w", encoding="utf-8") as f:
        f.write("".join(lines))

    return "".join(lines)


def run_checker(sysml_file, jar_path = "sysml.jar"):
    try:
        result = subprocess.run(
            ["java", "-jar", jar_path, sysml_file],
            capture_output=True,
            text=True
        )

        print("\n Checker Output:\n")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print("\n Checker Error:\n")
        print(e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    split = list(ds.keys())[0]   # "train"

    drop_sources = {"environment/alfworld"}
    ds_filtered = ds[split].filter(lambda x: x["source"] not in drop_sources)

    # Choix instance
    # 191
    # 225
    # 22
    # 505
    idx = 505
    instance = ds[split][idx]
    c_graph = {'nodes': {'1': 'action1.','2': 'action2.', '3': 'action3.','4': 'action3.', '5': 'action5.', '6': 'action6.','7': 'action7.', '8': 'action8.'},
                'edges': [('START', '1'), ('1', '2'), ('1', '5'), ('2', '3'), ('2', '4'), ('3', '8'), ('5', '4'),('4', '7'), ('5', '6'), ('6', '7'), ('7', '8'), ('8', 'END')]}

    # Extract nodes/edges
    graph = extract_nodes_and_edges(instance)
    if graph is None:
        raise RuntimeError("No assistant message found to extract nodes/edges.")

    # affichage
    print(format_graph(c_graph))

    # SysML
    sysml_code = transfrom_into_sysml(
        c_graph
    )

    print(sysml_code)
    # run_checker("sysml2.sysml")



