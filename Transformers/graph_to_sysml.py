from collections import defaultdict, deque

import subprocess
import sys


nodes = {}

def transform_into_sysml(graph: dict, package_name="PackageName", process_name="ProcessName"):

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
        for i in node_text.keys():
            if i not in succ.keys():
                succ[i] = []
            if i not in pred.keys():
                pred[i] = []

        return succ, pred

    # function that generates the SysML action definitions
    def action_def():
        action_defs = []
        # print(node_text)
        for node in sorted(node_text.keys(), key=sort_key):

            if len(pred[node]) > 1 or (len(pred[node]) > 0 and len(succ[pred[node][0]])) == 1:
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
            # added code
            else:
                for p in pred[node]:
                    if (len(pred[p]) == 0) and (p != 'START') and (p not in visited):
                        dfs(p)
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

    # with open("test1.sysml", "w", encoding="utf-8") as f:
    #     f.write("".join(lines))

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




