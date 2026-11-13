"""Fix SysML v2 naming problems in the enriched SOPBench models.

Reads  : enriched dataset/sopbench_nemotron_together.csv   (never modified)
Writes : enriched dataset/fixed/sopbench_nemotron_together.csv

Only the `sysml_model` column is changed:

1. Names that are SysML v2 reserved words (package, action, part, ... declarations) are suffixed with
   `_` (e.g. `package library` -> `package library_`), and every reference to the renamed name is
   updated (`library::X` -> `library_::X`, plain references too). String literals are never touched.
2. An instance with the same name as its type (`WorkflowAgent : WorkflowAgent;`) is renamed to the
   lower-camel-case form (`workflowAgent`), and `agent = WorkflowAgent;` is redirected to the instance.

No validation is performed here.
"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "enriched dataset" / "sopbench_nemotron_together.csv"
DST = Path(__file__).resolve().parent / "sopbench_nemotron_together.csv"

SYSML_RESERVED_NAMES = {
    # names flagged by the project
    "actor", "all", "default", "end", "entry", "event", "flow", "individual", "library", "new",
    "standard", "verification",
    # remaining SysML v2 / KerML keywords that could be produced by sanitising SOPBench names
    "abstract", "accept", "action", "alias", "allocate", "allocation", "analysis", "and", "as", "assert",
    "assign", "assume", "attribute", "bind", "binding", "block", "by", "calc", "case", "comment", "concern",
    "connect", "connection", "constraint", "decide", "def", "defined", "dependency", "derived", "do", "doc",
    "else", "enum", "exhibit", "exit", "expose", "false", "filter", "first", "for", "fork", "frame", "from",
    "hastype", "if", "implies", "import", "in", "include", "inout", "interface", "istype", "item", "join",
    "language", "loop", "merge", "message", "meta", "metadata", "nonunique", "not", "null", "objective",
    "occurrence", "of", "or", "ordered", "out", "package", "parallel", "part", "perform", "port", "private",
    "protected", "public", "readonly", "redefines", "ref", "references", "render", "rendering", "rep",
    "require", "requirement", "return", "satisfy", "send", "snapshot", "specializes", "stakeholder",
    "state", "subject", "subsets", "succession", "terminate", "then", "timeslice", "to", "transition",
    "true", "type", "until", "use", "variant", "variation", "verify", "via", "view", "viewpoint", "when",
    "while", "xor",
}

STRING_LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"')
DECLARATION = re.compile(
    r"^\s*(?:private\s+|public\s+|protected\s+)?"
    r"(?:package|(?:action|part|item|attribute|port|state|ref\s+part)(?:\s+def)?)\s+(\w+)",
    re.M,
)
SAME_NAME_INSTANCE = re.compile(r"^(\s*)(\w+)(\s*:\s*)(\2)(\s*;)", re.M)


def map_code(model: str, fn) -> str:
    """Applies fn only to the parts of the model outside string literals."""
    out, last = [], 0
    for m in STRING_LITERAL.finditer(model):
        out.append(fn(model[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(fn(model[last:]))
    return "".join(out)


def rename_reserved(model: str) -> tuple[str, list[str]]:
    code_only = "".join(STRING_LITERAL.split(model))
    bad = sorted({n for n in DECLARATION.findall(code_only) if n in SYSML_RESERVED_NAMES})
    for name in bad:
        # only names that are *declared* with a reserved word are renamed; all their references follow
        pattern = re.compile(rf"(?<![\w.]){re.escape(name)}(?!\w)")
        model = map_code(model, lambda s, p=pattern, n=name: p.sub(n + "_", s))
    return model, bad


def rename_same_name_instances(model: str) -> tuple[str, list[str]]:
    changed = []
    for m in list(SAME_NAME_INSTANCE.finditer(model)):
        type_name = m.group(2)
        new = type_name[0].lower() + type_name[1:]
        if new == type_name or re.search(rf"\b{re.escape(new)}\b", model):
            new = type_name + "Instance"
        model = SAME_NAME_INSTANCE.sub(
            lambda x, t=type_name, n=new: f"{x.group(1)}{n}{x.group(3)}{t}{x.group(5)}"
            if x.group(2) == t else x.group(0),
            model,
        )
        model = re.sub(rf"(\bagent\s*=\s*){re.escape(type_name)}(\s*;)", rf"\g<1>{new}\g<2>", model)
        changed.append(type_name)
    return model, changed


def fix_model(model: str) -> tuple[str, list[str]]:
    changes = []
    model, renamed = rename_reserved(model)
    changes += [f"reserved:{n}" for n in renamed]
    model, instances = rename_same_name_instances(model)
    changes += [f"instance:{n}" for n in dict.fromkeys(instances)]
    return model, changes


def main() -> int:
    csv.field_size_limit(sys.maxsize)
    with SRC.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        columns = [h.strip('\ufeff"\' ') for h in header]
        col = columns.index("sysml_model")
        rows = list(reader)

    changed_rows, tally = 0, {}
    for row in rows:
        row[col], changes = fix_model(row[col])
        changed_rows += bool(changes)
        for c in changes:
            tally[c] = tally.get(c, 0) + 1

    with DST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"rows={len(rows)} changed={changed_rows} {tally}\nwrote {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
