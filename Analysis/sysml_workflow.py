"""Syntax-aware reader/writer for the SysML v2 workflow subset produced by the three generators.

The three datasets use two control-flow dialects:

* explicit   (SAP-SAM, SOP-Bench): ``first a then b;``, ``decide d; first d; if g then x; else y;``
* chained    (WorFBench):          ``first start; then fork; then a; then b; action a {...} then j; join j;``

In the chained dialect a ``then`` statement that has no ``first`` takes the most recently *declared*
node (action, fork, join, decide or ``first X;`` target) as its source; ``then <name>;`` is only a
reference and never changes that source. Both dialects are read into one ``Workflow`` graph.

Anything outside the supported subset raises ``UnsupportedSyntax`` so that callers can flag the
workflow instead of guessing.
"""
from __future__ import annotations

import re
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

NODE_KEYWORDS = ("decide", "fork", "join", "merge")
TERMINALS = ("start", "done")


class UnsupportedSyntax(ValueError):
    pass


@dataclass
class Action:
    name: str
    type: str | None = None
    description: str | None = None  # raw string literal, quotes included
    description_op: str | None = None  # operator used in the source: "", ":>>", "::>", "attribute :>>"
    agent: str | None = None
    agent_op: str | None = None
    role_agent_assignments: list[tuple[str, str]] = field(default_factory=list)
    outputs: list[tuple[str, str | None]] = field(default_factory=list)


@dataclass
class Node:
    kind: str
    name: str
    anonymous: bool = False


@dataclass
class Workflow:
    imports: list[str] = field(default_factory=list)
    package: str = ""
    def_name: str = ""
    agent_types: list[tuple[str, str]] = field(default_factory=list)  # (name, parent)
    agent_insts: list[tuple[str, str]] = field(default_factory=list)  # (name, type)
    actions: dict[str, Action] = field(default_factory=dict)
    nodes: dict[str, Node] = field(default_factory=dict)
    # ordered flow items: ("edge", a, b) or ("decision", node)
    flow: list[tuple] = field(default_factory=list)
    decisions: dict[str, dict] = field(default_factory=dict)  # node -> {"branches": [(guard, target)], "else": t}
    standalone_first: list[str] = field(default_factory=list)
    duplicate_declarations: list[str] = field(default_factory=list)
    duplicate_control_nodes: list[tuple[str, str]] = field(default_factory=list)
    duplicate_agent_declarations: list[str] = field(default_factory=list)
    nested_defs: int = 0

    def edges(self) -> list[tuple[str, str]]:
        """Unique directed edges in first-appearance order (guards are kept in ``decisions``)."""
        seen, out = set(), []
        for item in self.flow:
            pairs = []
            if item[0] == "edge":
                pairs = [(item[1], item[2])]
            else:
                block = self.decisions[item[1]]
                pairs = [(item[1], t) for _, t in block["branches"]]
                if block["else"] is not None:
                    pairs.append((item[1], block["else"]))
            for p in pairs:
                if p not in seen:
                    seen.add(p)
                    out.append(p)
        return out


TOKEN_RE = re.compile(
    r"""\s+|//[^\n]*|/\*.*?\*/                       # whitespace / comments
    |(?P<str>"(?:[^"\\]|\\.)*")
    |(?P<op>::>|:>>|:>|::|==|!=|>=|<=)
    |(?P<id>[A-Za-z_]\w*)
    |(?P<num>\d+(?:\.\d+)?)
    |(?P<sym>[{};:=.,()\[\]<>!*+\-/%])
    |(?P<bad>.)""",
    re.X | re.S,
)


def tokenize(text: str) -> list[tuple[str, str]]:
    toks = []
    for m in TOKEN_RE.finditer(text):
        kind = m.lastgroup
        if kind is None:
            continue
        if kind == "bad":
            raise UnsupportedSyntax(f"unsupported character {m.group()!r}")
        toks.append((kind, m.group()))
    return toks


class _Parser:
    def __init__(self, text: str):
        self.t = tokenize(text)
        self.i = 0
        self.wf = Workflow()
        self.prev: str | None = None
        self.pending: str | None = None
        self.anon = 0

    # token helpers -----------------------------------------------------------------------
    def peek(self, k: int = 0) -> str | None:
        j = self.i + k
        return self.t[j][1] if j < len(self.t) else None

    def next(self) -> str:
        if self.i >= len(self.t):
            raise UnsupportedSyntax("unexpected end of model")
        v = self.t[self.i][1]
        self.i += 1
        return v

    def expect(self, v: str) -> None:
        got = self.next()
        if got != v:
            raise UnsupportedSyntax(f"expected {v!r}, found {got!r}")

    def ident(self) -> str:
        kind = self.t[self.i][0] if self.i < len(self.t) else None
        if kind != "id":
            raise UnsupportedSyntax(f"expected identifier near {self.peek()!r}")
        return self.next()

    def qname(self) -> str:
        parts = [self.ident()]
        while self.peek() == "::" and self.peek(1) != "*":
            self.next()
            parts.append(self.ident())
        return "::".join(parts)

    # grammar -----------------------------------------------------------------------------
    def parse(self) -> Workflow:
        wf = self.wf
        while self.peek() in ("private", "import"):
            if self.peek() == "private":
                self.next()
            self.expect("import")
            q = self.qname()
            self.expect("::")
            self.expect("*")
            self.expect(";")
            wf.imports.append(q + "::*")
        self.expect("package")
        wf.package = self.ident()
        self.expect("{")
        while self.peek() != "}":
            self.expect("action")
            self.expect("def")
            if wf.def_name:
                wf.nested_defs += 1
                raise UnsupportedSyntax("more than one action def")
            wf.def_name = self.ident()
            self.expect("{")
            self.body()
            self.expect("}")
        self.expect("}")
        if self.i != len(self.t):
            raise UnsupportedSyntax("content after package")
        if not wf.def_name:
            raise UnsupportedSyntax("no action def")
        return wf

    def declare(self, store: dict, key: str, value) -> None:
        if key in store:
            self.wf.duplicate_declarations.append(key)
        else:
            store[key] = value

    def body(self) -> None:
        wf = self.wf
        while self.peek() != "}":
            w = self.peek()
            if w is None:
                raise UnsupportedSyntax("unterminated action def")
            if w == "part":
                self.next()
                self.expect("def")
                name = self.ident()
                self.expect("specializes")
                parent = self.qname()
                if self.peek() == ";":
                    self.next()
                else:
                    self.expect("{")
                    self.expect("}")
                if (name, parent) not in wf.agent_types:
                    wf.agent_types.append((name, parent))
                else:
                    wf.duplicate_agent_declarations.append(name)
            elif w == "action":
                self.next()
                a = self.action()
                self.declare(wf.actions, a.name, a)
                self.prev = a.name
            elif w in NODE_KEYWORDS:
                self.next()
                n = self.node(w)
                self.prev = n.name
            elif w == "first":
                self.next()
                a = self.ident_or_terminal()
                if self.peek() == "then":
                    self.next()
                    b = self.ident_or_terminal()
                    self.expect(";")
                    wf.flow.append(("edge", a, b))
                else:
                    self.expect(";")
                    self.prev = self.pending = a
                    wf.standalone_first.append(a)
            elif w == "then":
                self.next()
                self.then_statement()
            elif w == "if":
                self.next()
                guard = []
                while self.peek() != "then":
                    if self.peek() in (None, ";", "{", "}"):
                        raise UnsupportedSyntax("malformed guard")
                    guard.append(self.next())
                self.next()
                target = self.ident_or_terminal()
                self.expect(";")
                self.block()["branches"].append((self.join_guard(guard), target))
            elif w == "else":
                self.next()
                if self.peek() == "if":
                    raise UnsupportedSyntax("else-if is not supported")
                target = self.ident_or_terminal()
                self.expect(";")
                blk = self.block()
                if blk["else"] is not None:
                    raise UnsupportedSyntax("two else branches")
                blk["else"] = target
            elif self.t[self.i][0] == "id" and self.peek(1) == ":":
                name = self.ident()
                self.next()
                typ = self.qname()
                self.expect(";")
                if (name, typ) not in wf.agent_insts:
                    wf.agent_insts.append((name, typ))
                else:
                    wf.duplicate_agent_declarations.append(name)
            else:
                raise UnsupportedSyntax(f"unsupported statement starting with {w!r}")

    @staticmethod
    def join_guard(tokens: list[str]) -> str:
        out = ""
        for tok in tokens:
            if out and (tok[0].isalnum() or tok[0] in "_\"") and (out[-1].isalnum() or out[-1] in "_\""):
                out += " "
            elif out and tok in ("==", "!=", ">=", "<=", "<", ">"):
                out += " "
            elif out and out[-1] in "=<>" and tok[0] not in "=":
                out += " "
            out += tok
        return out

    def ident_or_terminal(self) -> str:
        return self.ident()

    def block(self) -> dict:
        if self.pending is None:
            raise UnsupportedSyntax("if/else without a preceding 'first <node>;'")
        node = self.pending
        self.wf.standalone_first = [n for n in self.wf.standalone_first if n != node]
        if node not in self.wf.decisions:
            self.wf.decisions[node] = {"branches": [], "else": None}
            self.wf.flow.append(("decision", node))
        return self.wf.decisions[node]

    def node(self, kind: str) -> Node:
        anonymous = self.peek() == ";"
        if anonymous:
            self.anon += 1
            name = f"__anon_{kind}_{self.anon}"
        else:
            name = self.ident()
        self.expect(";")
        n = Node(kind, name, anonymous)
        if name in self.wf.nodes:
            if self.wf.nodes[name].kind == kind:
                self.wf.duplicate_control_nodes.append((kind, name))
            else:
                self.wf.duplicate_declarations.append(name)
        else:
            self.wf.nodes[name] = n
        return n

    def then_statement(self) -> None:
        if self.prev is None:
            raise UnsupportedSyntax("'then' without a source")
        w = self.peek()
        if w == "action":
            self.next()
            if self.t[self.i][0] == "id" and self.peek(1) == ";":
                # WorFBench writes a branch-target reference as ``then actionX;`` after a
                # fork. It is not a declaration and must not advance the fork's branch source.
                target = self.ident()
                self.expect(";")
                self.wf.flow.append(("edge", self.prev, target))
                return
            a = self.action()
            self.declare(self.wf.actions, a.name, a)
            self.wf.flow.append(("edge", self.prev, a.name))
            self.prev = a.name
        elif w in NODE_KEYWORDS:
            self.next()
            n = self.node(w)
            self.wf.flow.append(("edge", self.prev, n.name))
            self.prev = n.name
        else:
            target = self.ident_or_terminal()
            self.expect(";")
            self.wf.flow.append(("edge", self.prev, target))

    def action(self) -> Action:
        a = Action(self.ident())
        if self.peek() == ":":
            self.next()
            a.type = self.qname()
        if self.peek() == ";":
            self.next()
            return a
        self.expect("{")
        while self.peek() != "}":
            op = ""
            if self.peek() in ("::>", ":>>"):
                op = self.next()
            elif self.peek() == "attribute":
                self.next()
                op = "attribute"
                if self.peek() == ":>>":
                    op = "attribute :>>"
                    self.next()
            w = self.peek()
            if w == "description":
                self.next()
                self.expect("=")
                if self.t[self.i][0] != "str":
                    raise UnsupportedSyntax("description is not a string literal")
                if a.description is not None:
                    raise UnsupportedSyntax("two descriptions")
                a.description, a.description_op = self.next(), op
                self.expect(";")
            elif w == "agent" and op in ("", ":>>"):
                self.next()
                self.expect("=")
                if a.agent is not None:
                    raise UnsupportedSyntax("two agent assignments")
                a.agent, a.agent_op = self.ident(), op
                self.expect(";")
            elif w is not None and self.t[self.i][0] == "id" and op in ("", ":>>") and self.peek(1) == "=":
                role = self.ident()
                self.expect("=")
                instance = self.ident()
                self.expect(";")
                a.role_agent_assignments.append((role, instance))
            elif w == "out" and not op:
                self.next()
                name = self.ident()
                typ = None
                if self.peek() == ":":
                    self.next()
                    typ = self.qname()
                self.expect(";")
                a.outputs.append((name, typ))
            else:
                raise UnsupportedSyntax(f"unsupported action member {w!r} in {a.name}")
        self.expect("}")
        return a


def parse_workflow(text: str) -> Workflow:
    wf = _Parser(text).parse()
    # Name anonymous nodes after the order of appearance, avoiding collisions.
    taken = set(wf.actions) | set(wf.nodes) | {n for n, _ in wf.agent_insts} | {n for n, _ in wf.agent_types}
    rename: dict[str, str] = {}
    counters: dict[str, int] = {}
    for name, node in list(wf.nodes.items()):
        if node.anonymous:
            while True:
                counters[node.kind] = counters.get(node.kind, 0) + 1
                cand = f"{node.kind}{counters[node.kind]}"
                if cand not in taken:
                    break
            taken.add(cand)
            rename[name] = cand
    if rename:
        wf.nodes = {rename.get(k, k): Node(v.kind, rename.get(k, k), v.anonymous) for k, v in wf.nodes.items()}
        wf.flow = [
            ("edge", rename.get(i[1], i[1]), rename.get(i[2], i[2])) if i[0] == "edge" else ("decision", rename.get(i[1], i[1]))
            for i in wf.flow
        ]
        wf.standalone_first = [rename.get(n, n) for n in wf.standalone_first]
        wf.decisions = {
            rename.get(k, k): {
                "branches": [(g, rename.get(t, t)) for g, t in v["branches"]],
                "else": rename.get(v["else"], v["else"]) if v["else"] is not None else None,
            }
            for k, v in wf.decisions.items()
        }
    return wf


def render_workflow(wf: Workflow, description_op: str = "::>") -> str:
    """Canonical MOSAICO layout: imports, agents, actions, control nodes, then control flow."""
    pad = "        "
    out = [f"private import {i};" for i in wf.imports] + ["", f"package {wf.package} {{", "", f"    action def {wf.def_name} {{", ""]
    for name, parent in wf.agent_types:
        out += [f"{pad}part def {name} specializes {parent} {{", f"{pad}}}", ""]
    for name, typ in wf.agent_insts:
        out.append(f"{pad}{name} : {typ};")
    if wf.agent_insts:
        out.append("")
    for a in wf.actions.values():
        out.append(f"{pad}action {a.name} : Step {{")
        out.append(f"{pad}    {description_op} description = {a.description};")
        out.append(f"{pad}    agent = {a.agent};")
        for oname, otype in a.outputs:
            out.append(f"{pad}    out {oname}" + (f" : {otype};" if otype else ";"))
        out += [f"{pad}}}", ""]
    for n in wf.nodes.values():
        out.append(f"{pad}{n.kind} {n.name};")
    if wf.nodes:
        out.append("")
    for node in wf.standalone_first:
        out.append(f"{pad}first {node};")
    if wf.standalone_first:
        out.append("")
    for item in wf.flow:
        if item[0] == "edge":
            out.append(f"{pad}first {item[1]} then {item[2]};")
        else:
            blk = wf.decisions[item[1]]
            out.append(f"{pad}first {item[1]};")
            out += [f"{pad}    if {g} then {t};" for g, t in blk["branches"]]
            if blk["else"] is not None:
                out.append(f"{pad}    else {blk['else']};")
    out += ["    }", "}", ""]
    return "\n".join(out)


BASE_AGENT_TYPES = {
    "Agent", "SolutionAgent", "SupervisionAgent", "ReferenceAgent",
    "ConsensusAgent",
}


def _boolean_guard_outputs(wf: Workflow) -> set[tuple[str, str]]:
    used = set()
    for block in wf.decisions.values():
        for guard, _ in block["branches"]:
            refs = re.findall(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b", guard)
            for action_name, output_name in refs:
                ref = re.escape(f"{action_name}.{output_name}")
                reduced = re.sub(rf"\b{ref}\b", "VALUE", guard)
                atoms = re.split(r"\b(?:and|or)\b", reduced, flags=re.I)
                if any(
                    re.fullmatch(
                        r"\s*(?:not\s+)?VALUE(?:\s*(?:==|!=)\s*(?:true|false))?\s*",
                        atom, re.I,
                    )
                    for atom in atoms
                ):
                    used.add((action_name, output_name))
    return used


def _topology(wf: Workflow) -> tuple:
    return (
        tuple(sorted(wf.actions)),
        tuple(sorted((n, x.kind) for n, x in wf.nodes.items())),
        tuple(sorted(wf.edges())),
        tuple(sorted((n, tuple(sorted(set(v["branches"]))), v["else"]) for n, v in wf.decisions.items())),
        tuple(sorted(wf.standalone_first)),
    )


def normalize_workflow(text: str) -> tuple[str, list[str], Workflow]:
    """Normalize the supported generator subset without guessing missing behavior.

    Returns (canonical SysML, transformations, parsed canonical model). Unsupported or ambiguous
    models raise ``UnsupportedSyntax`` and must be excluded by the caller.
    """
    wf = parse_workflow(text)
    original_topology = _topology(wf)
    if wf.duplicate_declarations:
        raise UnsupportedSyntax(f"duplicate declarations: {', '.join(wf.duplicate_declarations)}")
    if wf.duplicate_control_nodes:
        transformations = ["removed identical duplicate control-node declarations"]
    else:
        transformations = []
    if wf.duplicate_agent_declarations:
        wf.agent_types = list(dict.fromkeys(wf.agent_types))
        wf.agent_insts = list(dict.fromkeys(wf.agent_insts))
        transformations.append("removed identical duplicate agent declarations")
    if not wf.actions:
        raise UnsupportedSyntax("workflow contains no actions")

    wanted_imports = ["MOSAICO::*", "ScalarValues::*"]
    seen_imports = set(wanted_imports)
    for imp in wf.imports:
        if imp not in seen_imports:
            wanted_imports.append(imp)
            seen_imports.add(imp)
    if wf.imports != wanted_imports:
        transformations.append("normalized required imports and removed duplicates")
    wf.imports = wanted_imports

    # Names share the control-flow namespace; reject collisions rather than silently renaming.
    node_names = set(wf.actions) | set(wf.nodes)
    if len(node_names) != len(wf.actions) + len(wf.nodes):
        raise UnsupportedSyntax("action and control-node names are not unique")
    agent_names = [n for n, _ in wf.agent_types] + [n for n, _ in wf.agent_insts]
    if len(agent_names) != len(set(agent_names)):
        raise UnsupportedSyntax("agent declaration names are not unique")
    if set(agent_names) & node_names:
        raise UnsupportedSyntax("agent and workflow node names collide")

    for action in wf.actions.values():
        if not action.description:
            raise UnsupportedSyntax(f"action {action.name} has no description")
        if action.type != "Step":
            action.type = "Step"
            transformations.append(f"added Step specialization to {action.name}")
        if action.description_op != "::>":
            action.description_op = "::>"
            transformations.append(f"normalized description operator on {action.name}")

        legacy_instances = {instance for _, instance in action.role_agent_assignments}
        if len(legacy_instances) > 1:
            raise UnsupportedSyntax(f"action {action.name} has ambiguous legacy agent assignments")
        if legacy_instances:
            legacy_instance = next(iter(legacy_instances))
            if action.agent and action.agent != legacy_instance:
                raise UnsupportedSyntax(f"action {action.name} has conflicting agent assignments")
            action.agent = legacy_instance
            transformations.append(f"normalized legacy agent-role assignment on {action.name}")

    # An untyped output is inferred only when a guard requires a Boolean or the prose explicitly
    # states that the output is true/false. All other cases are left for manual review.
    boolean_guard_outputs = _boolean_guard_outputs(wf)
    for action in wf.actions.values():
        replaced = []
        for name, typ in action.outputs:
            if typ is None:
                if (action.name, name) in boolean_guard_outputs or re.search(
                    r"\b(?:returns?|output(?:s)?)\b[^.]{0,100}\btrue\b[^.]{0,100}\bfalse\b",
                    action.description or "", re.I,
                ):
                    typ = "Boolean"
                    transformations.append(f"inferred Boolean type for {action.name}.{name}")
                else:
                    raise UnsupportedSyntax(f"cannot infer type for output {action.name}.{name}")
            replaced.append((name, typ))
        action.outputs = replaced

    declared_agent_types = {name for name, _ in wf.agent_types}
    declared_instances = {name: typ for name, typ in wf.agent_insts}
    if not wf.agent_insts:
        if "solutionAgent" in node_names or "solutionAgent" in declared_agent_types:
            raise UnsupportedSyntax("solutionAgent default name collides with another declaration")
        wf.agent_insts.append(("solutionAgent", "SolutionAgent"))
        declared_instances["solutionAgent"] = "SolutionAgent"
        transformations.append("added default solutionAgent instance")
    unresolved_agent_types = {
        typ for typ in declared_instances.values()
        if typ not in declared_agent_types and typ not in BASE_AGENT_TYPES
    }
    if unresolved_agent_types:
        raise UnsupportedSyntax(f"agent instance types are undeclared: {', '.join(sorted(unresolved_agent_types))}")
    missing = [a.name for a in wf.actions.values() if not a.agent]
    invalid = [(a.name, a.agent) for a in wf.actions.values()
               if a.agent and a.agent not in declared_instances]
    if invalid:
        raise UnsupportedSyntax("actions refer to undeclared agent instances: " +
                                ", ".join(f"{a}->{g}" for a, g in invalid))
    if missing:
        existing = sorted(declared_instances)
        fallback = existing[0] if len(existing) == 1 else None
        if fallback is None:
            if "solutionAgent" in declared_instances:
                fallback = "solutionAgent"
            elif "solutionAgent" in node_names or "solutionAgent" in declared_agent_types:
                raise UnsupportedSyntax("solutionAgent fallback collides with a declaration")
            else:
                wf.agent_insts.append(("solutionAgent", "SolutionAgent"))
                declared_instances["solutionAgent"] = "SolutionAgent"
                fallback = "solutionAgent"
                transformations.append("added default solutionAgent for unassigned actions")
        for name in missing:
            wf.actions[name].agent = fallback
            transformations.append(f"assigned {name} to {fallback}")

    for action in wf.actions.values():
        if not action.agent or action.agent not in declared_instances:
            raise UnsupportedSyntax(f"action {action.name} has no valid agent assignment")

    allowed = set(wf.actions) | set(wf.nodes) | {"start", "done"}
    for node in wf.standalone_first:
        if node not in allowed:
            raise UnsupportedSyntax(f"first statement references undeclared node {node}")
    for source, target in wf.edges():
        if source not in allowed or target not in allowed:
            raise UnsupportedSyntax(f"control-flow edge has undeclared endpoint: {source} -> {target}")
        if source == "done":
            raise UnsupportedSyntax("done has outgoing control flow")
    for decision, block in wf.decisions.items():
        if decision not in wf.nodes or wf.nodes[decision].kind != "decide":
            raise UnsupportedSyntax(f"decision block {decision} has no decide declaration")
        for guard, _ in block["branches"]:
            refs = re.findall(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b", guard)
            if not refs:
                raise UnsupportedSyntax(f"decision {decision} guard has no declared output reference")
            for action_name, output_name in refs:
                action = wf.actions.get(action_name)
                if action is None:
                    raise UnsupportedSyntax(f"guard references undeclared action {action_name}")
                types = {typ for name, typ in action.outputs if name == output_name}
                if not types:
                    raise UnsupportedSyntax(f"guard references undeclared output {action_name}.{output_name}")
                if (action_name, output_name) in boolean_guard_outputs and types != {"Boolean"}:
                    raise UnsupportedSyntax(f"guard output {action_name}.{output_name} is not Boolean")

    # Duplicate graph arcs do not encode distinct SysML action occurrences and add no behavior;
    # retain one arc deterministically and record the cleanup.
    before = len(wf.flow)
    unique_flow = []
    seen_arcs = set()
    for item in wf.flow:
        if item[0] == "edge":
            arc = (item[1], item[2])
            if arc in seen_arcs:
                continue
            seen_arcs.add(arc)
        unique_flow.append(item)
    wf.flow = unique_flow
    branch_duplicates = False
    for block in wf.decisions.values():
        branches = list(dict.fromkeys(block["branches"]))
        branch_duplicates |= len(branches) != len(block["branches"])
        block["branches"] = branches
    if len(wf.flow) < before or branch_duplicates:
        transformations.append("removed duplicate control-flow edges")

    normalized = render_workflow(wf)
    parsed = parse_workflow(normalized)
    if original_topology != _topology(parsed):
        raise UnsupportedSyntax("normalization changed control-flow topology")
    transformations.append("verified original/normalized control-flow topology")
    return normalized, transformations, parsed


def extract_complexity_features(wf: Workflow, source: str = "") -> dict:
    """Return interpretable graph features; path depth is null for cyclic graphs."""
    edges = wf.edges()
    graph: dict[str, set[str]] = defaultdict(set)
    all_nodes = set(wf.actions) | set(wf.nodes) | {"start", "done"}
    for a, b in edges:
        graph[a].add(b)
        all_nodes.update((a, b))
    indegree = {n: 0 for n in all_nodes}
    for a in graph:
        for b in graph[a]:
            indegree[b] = indegree.get(b, 0) + 1
    forks = [n for n, x in wf.nodes.items() if x.kind == "fork"]
    joins = {n for n, x in wf.nodes.items() if x.kind == "join"}
    branch_counts = [len(graph.get(n, ())) for n in forks]
    conditional_count = sum(len(x["branches"]) + (x["else"] is not None) for x in wf.decisions.values())

    # Kahn topological traversal gives cycle detection and longest executable path in one pass.
    original_indegree = indegree.copy()
    queue = deque(sorted(n for n in all_nodes if indegree.get(n, 0) == 0))
    topo = []
    dist = {n: 1 if n in wf.actions else 0 for n in all_nodes}
    while queue:
        n = queue.popleft()
        topo.append(n)
        for nxt in sorted(graph.get(n, ())):
            weight = 1 if nxt in wf.actions else 0
            dist[nxt] = max(dist[nxt], dist[n] + weight)
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    cyclic = len(topo) < len(all_nodes)
    max_depth = None if cyclic else max(dist.values(), default=0)

    def reachable(start: str) -> set[str]:
        seen, todo = set(), [start]
        while todo:
            current = todo.pop()
            if current in seen:
                continue
            seen.add(current)
            todo.extend(graph.get(current, ()))
        return seen

    def reachable_until(start: str, stop: str) -> set[str]:
        seen, todo = set(), [start]
        while todo:
            current = todo.pop()
            if current in seen:
                continue
            seen.add(current)
            if current != stop:
                todo.extend(graph.get(current, ()))
        return seen

    fork_regions: dict[str, set[str]] = {}
    for fork in forks:
        succ = sorted(graph.get(fork, ()))
        if len(succ) < 2:
            fork_regions[fork] = set()
            continue
        reach_by_branch = [reachable(s) for s in succ]
        common_joins = set.intersection(*(r & joins for r in reach_by_branch)) if reach_by_branch else set()
        if common_joins:
            distances = []
            for join in common_joins:
                dists = []
                for s in succ:
                    todo = deque([(s, 0)])
                    seen = set()
                    found = None
                    while todo:
                        node, depth = todo.popleft()
                        if node in seen:
                            continue
                        seen.add(node)
                        if node == join:
                            found = depth
                            break
                        todo.extend((n, depth + 1) for n in graph.get(node, ()))
                    dists.append(found if found is not None else math.inf)
                distances.append((max(dists), sum(dists), join))
            merge = min(distances)[2]
            fork_regions[fork] = set.union(
                *(reachable_until(successor, merge) for successor in succ)
            ) - {merge}
        else:
            fork_regions[fork] = set()
    nested = {f: {g for g in forks if f != g and g in fork_regions[f]} for f in forks}

    def nesting_depth(fork: str, path: set[str]) -> int:
        if fork in path:
            return 1
        children = nested.get(fork, set()) - path
        return 1 + max((nesting_depth(x, path | {fork}) for x in children), default=0)

    max_nesting = max((nesting_depth(f, set()) for f in forks), default=0)
    # A cycle means an action is revisited; incoming edges on their own do not imply repetition.
    repeated_actions = 0
    for action in wf.actions:
        pending = list(graph.get(action, ()))
        seen = set()
        repeats = False
        while pending:
            current = pending.pop()
            if current == action:
                repeats = True
                break
            if current in seen:
                continue
            seen.add(current)
            pending.extend(graph.get(current, ()))
        repeated_actions += int(repeats)

    # Duplicate action bodies are tracked separately for diagnosis; they are not execution events.
    quoted_removed = TOKEN_RE.sub(lambda m: " " * len(m.group()) if m.lastgroup == "str" else m.group(), source or "")
    decls = re.findall(
        r"\baction\s+(?!def\b)([A-Za-z_]\w*)"
        r"(?:\s*:\s*[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)?\s*\{",
        quoted_removed,
    )
    duplicate_declarations = sum(max(0, count - 1) for count in Counter(decls).values())
    return {
        "num_actions": len(wf.actions),
        "num_decisions": sum(n.kind == "decide" for n in wf.nodes.values()),
        "num_forks": len(forks),
        "num_joins": len(joins),
        "control_flow_edges": len(edges),
        "conditional_branches": conditional_count,
        "parallel_branches": sum(x for x in branch_counts if x > 1),
        "max_nested_parallelism_depth": max_nesting,
        "has_parallelism": any(x > 1 for x in branch_counts),
        "has_nested_parallelism": max_nesting > 1,
        "has_cycles": cyclic,
        "repeated_action_occurrences": repeated_actions,
        "duplicate_action_declarations": duplicate_declarations,
        "max_execution_path_depth": max_depth,
        "max_incoming_edges": max(original_indegree.values(), default=0),
    }


def validate_workflow_structure(wf: Workflow, source: str = "") -> dict:
    """Check executable entry, action reachability, and reachable termination.

    An explicit ``RAG-STRUCTURAL-EXCEPTION: ...`` source comment is recorded as an
    intentional exception, but does not make the workflow eligible for the RAG corpus.
    """
    edges = wf.edges()
    graph: dict[str, set[str]] = defaultdict(set)
    reverse: dict[str, set[str]] = defaultdict(set)
    for start, end in edges:
        graph[start].add(end)
        reverse[end].add(start)

    def reachable(root: str, adjacency: dict[str, set[str]]) -> set[str]:
        seen: set[str] = set()
        pending = [root]
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            pending.extend(adjacency.get(node, ()))
        return seen

    from_start = reachable("start", graph)
    to_done = reachable("done", reverse)
    errors = []
    if wf.actions and not edges:
        errors.append("declared actions have no executable control-flow edges")
    if not graph.get("start"):
        errors.append("start has no outgoing entry connection")

    unreachable_actions = sorted(set(wf.actions) - from_start)
    if unreachable_actions:
        errors.append("actions unreachable from start: " + ", ".join(unreachable_actions))

    if "done" not in from_start:
        errors.append("no termination is reachable from start")
    dead_end_actions = sorted(
        action for action in set(wf.actions) & from_start if action not in to_done
    )
    if dead_end_actions:
        errors.append("reachable actions have no path to done: " + ", ".join(dead_end_actions))

    exception_match = re.search(
        r"(?m)//\s*RAG-STRUCTURAL-EXCEPTION:\s*(.+?)\s*$", source
    )
    intentional_exception = exception_match.group(1).strip() if exception_match else ""
    structural_valid = not errors and not intentional_exception
    if intentional_exception:
        status = "intentional_exception"
    elif errors:
        status = "incomplete"
    else:
        status = "complete"
    return {
        "structural_valid": structural_valid,
        "structural_status": status,
        "structural_errors": errors,
        "intentional_exception": intentional_exception,
    }
