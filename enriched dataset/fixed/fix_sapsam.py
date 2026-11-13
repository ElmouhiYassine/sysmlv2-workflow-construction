"""Fix SysML v2 reserved-word names in the enriched SAP-SAM models.

Reads  : enriched dataset/sapsam_nemotron_together.csv   (never modified)
Writes : enriched dataset/fixed/sapsam_nemotron_together.csv

Only the `sysml_model` column is changed.

1. Broken description quoting: the generator emitted descriptions such as
   `"Click \\"GO\\"."GO\\""` (a closed literal followed by a duplicated/extra fragment containing a
   stray `"`). The text up to the first unescaped `"` is kept; the trailing fragment is dropped when it
   only repeats text already present, otherwise appended. Inner quotes become `'` (the quote characters are kept as single quotes).
   Lone backslashes in descriptions (`%USERPROFILE%\\Documents`) become `/`.
   Non-ASCII letters in identifiers (`getUmowaODziełoDraft`) are transliterated (`...Dzielo...`).
2. Same-name instance: `doctor : doctor;` is renamed `doctorInstance : doctor;` and `agent = doctor;`
   follows.
3. Reserved names: a name that is declared with a SysML v2 reserved word
(e.g. `action end : Step {`) is suffixed with `_` (`end_`) and every reference to it is updated
(`then end;` -> `then end_;`). Only name positions are rewritten (declaration, first/then/else,
`agent =`, `name.member`, `::`), so keyword uses such as `decide node1;` or `join node2;` are preserved. String literals (descriptions) are never touched, and a name that is
not declared in the model is left alone, so keyword uses of the word are not affected.

No validation is performed here.
"""
import csv
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "enriched dataset" / "sapsam_nemotron_together.csv"
DST = Path(__file__).resolve().parent / "sapsam_nemotron_together.csv"

SYSML_RESERVED_NAMES = {
    "actor",
    "all",
    "default",
    "end",
    "entry",
    "event",
    "flow",
    "individual",
    "library",
    "new",
    "standard",
    "verification",
    # found as action names in the SAP-SAM validation run
    "analysis",
    "decide",
    "transition",
    "verify",
    "rendering",
    "join",
    "accept",
    "exit",
    "terminate",
    "send",
    # remaining SysML v2 / KerML keywords
    "locale",
    # remaining KerML keywords
    "about", "assoc", "behavior", "bool", "chains", "class", "classifier", "composite", "conjugate",
    "conjugates", "conjugation", "connector", "const", "crosses", "datatype", "disjoining", "disjoint",
    "feature", "featured", "featuring", "function", "inv", "inverse", "inverting", "multiplicity",
    "namespace", "portion", "predicate", "redefinition", "specialization", "step", "struct",
    "subclassifier", "subsetting", "subtype", "typed", "typing", "var", "abstract", "expr", "member",
    "intersects", "differences", "unions", "textual", "dependency", "language", "rep",
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
# `action end : Step {`, `part def end ...`, `package library {`
DECLARATION = re.compile(
    r"^\s*(?:private\s+|public\s+|protected\s+)?"
    r"(?:package|(?:action|part|item|attribute|port|state)(?:\s+def)?|ref\s+part)\s+(\w+)",
    re.M,
)
DESCRIPTION_LINE = re.compile(r'^(\s*(?::>>\s*)?description\s*=\s*)"(.*)";[ \t]*$')
# `end : Type;` (instance declaration)
INSTANCE = re.compile(r"^\s*(\w+)\s*:\s*[\w:]+\s*;", re.M)


def map_code(model: str, fn) -> str:
    """Applies fn only to the parts of the model that are outside string literals."""
    out, last = [], 0
    for m in STRING_LITERAL.finditer(model):
        out.append(fn(model[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(fn(model[last:]))
    return "".join(out)


def fix_description(inner: str) -> str | None:
    """Returns the repaired text, or None if the literal is already well formed."""
    m = re.match(r'((?:[^"\\]|\\.)*)"(.*)$', inner)
    if not m:
        return None
    head, tail = m.group(1), m.group(2)
    plain = lambda t: t.replace('\\"', "'").replace('"', "'")
    head_text, tail_text = plain(head), plain(tail)
    text = head_text
    if tail_text.strip() and tail_text.strip() not in head_text:
        text = f"{head_text} {tail_text}".strip()
    return text.replace("\\", "/")


def fix_quotes(model: str) -> tuple[str, int]:
    lines, count = model.split("\n"), 0
    for i, line in enumerate(lines):
        m = DESCRIPTION_LINE.match(line)
        if not m:
            continue
        fixed = fix_description(m.group(2))
        if fixed is not None:
            lines[i] = f'{m.group(1)}"{fixed}";'
            count += 1
    return "\n".join(lines), count


PART_DEF = re.compile(r"^[ \t]*part[ \t]+def[ \t]+(\w+)", re.M)
ACTION_OR_NODE = re.compile(r"^[ \t]*(?:action|fork|join|decide|merge)[ \t]+(\w+)", re.M)
INSTANCE_DECL = re.compile(r"^[ \t]*(\w+)[ \t]*:[ \t]*(\w+)[ \t]*;", re.M)


def fix_agent_collisions(model: str) -> tuple[str, list[str]]:
    """Agent part defs / instances sharing a name with an action or node are renamed with a `_` suffix.

    Instances are only referenced through `agent = name;`, part defs through `specializes` / `: Type;`,
    so actions and `first/then` references keep their name.
    """
    code = "".join(STRING_LITERAL.split(model))
    taken = set(re.findall(r"\w+", code))
    action_names = set(ACTION_OR_NODE.findall(code))
    changed = []
    for name in sorted(set(INSTANCE_DECL.findall(code) and [m[0] for m in INSTANCE_DECL.findall(code)]) & action_names):
        new = name + "_"
        while new in taken:
            new += "_"
        taken.add(new)
        model = INSTANCE_DECL.sub(lambda m, n=name, w=new: m.group(0).replace(n, w, 1) if m.group(1) == n else m.group(0), model)
        ref = re.compile(rf"(\bagent\s*=\s*){re.escape(name)}(?=\s*;)")
        model = map_code(model, lambda t, r=ref, w=new: r.sub(rf"\g<1>{w}", t))
        changed.append(f"{name}:instance-vs-action")
    code = "".join(STRING_LITERAL.split(model))
    for name in sorted(set(PART_DEF.findall(code)) & set(ACTION_OR_NODE.findall(code))):
        new = name + "_"
        while new in taken:
            new += "_"
        taken.add(new)
        decl = re.compile(rf"^([ \t]*part[ \t]+def[ \t]+){re.escape(name)}(?!\w)", re.M)
        typ = re.compile(rf"^([ \t]*\w+[ \t]*:[ \t]*){re.escape(name)}([ \t]*;)", re.M)
        model = decl.sub(rf"\g<1>{new}", model)
        model = typ.sub(rf"\g<1>{new}\g<2>", model)
        changed.append(f"{name}:partdef-vs-action")
    return model, changed


SAME_NAME_INSTANCE = re.compile(r"^([ \t]*)(\w+)([ \t]*:[ \t]*)\2([ \t]*;)", re.M)


def fix_same_name_instances(model: str) -> tuple[str, list[str]]:
    """`doctor : doctor;` -> `doctorInstance : doctor;` and `agent = doctor;` -> `agent = doctorInstance;`."""
    code_only = "".join(STRING_LITERAL.split(model))
    names = sorted({m.group(2) for m in SAME_NAME_INSTANCE.finditer(code_only)})
    for name in names:
        new = f"{name}Instance"
        if re.search(rf"(?<![\w.]){re.escape(new)}(?!\w)", code_only):
            new = f"{name}Instance1"
        model = SAME_NAME_INSTANCE.sub(
            lambda m, n=name, w=new: f"{m.group(1)}{w}{m.group(3)}{n}{m.group(4)}" if m.group(2) == n else m.group(0),
            model,
        )
        ref = re.compile(rf"(\bagent\s*=\s*){re.escape(name)}(?=\s*;)")
        model = map_code(model, lambda t, r=ref, w=new: r.sub(rf"\g<1>{w}", t))
    return model, [f"{n}:same-name-instance" for n in names]


SPECIAL = {
    "ß": "ss", "æ": "ae", "Æ": "Ae", "œ": "oe", "Œ": "Oe", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D",
    "ð": "d", "Ð": "D", "þ": "th", "Þ": "Th", "ł": "l", "Ł": "L", "ı": "i", "ħ": "h", "Ħ": "H",
}


def to_ascii(text: str) -> str:
    out = []
    for ch in text:
        if ord(ch) < 128:
            out.append(ch)
        elif ch in SPECIAL:
            out.append(SPECIAL[ch])
        else:
            base = unicodedata.normalize("NFKD", ch)
            out.append("".join(c for c in base if ord(c) < 128 and not unicodedata.combining(c)))
    return "".join(out)


def fix_backslashes(model: str) -> tuple[str, int]:
    """A lone backslash in a description (`%USERPROFILE%\\Documents`, `Callguide\\Survey`) is an invalid
    escape in a SysML string; it is replaced by `/`. `%` and other symbols are legal and kept."""
    lines, count = model.split("\n"), 0
    for i, line in enumerate(lines):
        m = DESCRIPTION_LINE.match(line)
        if m and "\\" in m.group(2):
            lines[i] = f'{m.group(1)}"{m.group(2).replace(chr(92), "/")}";'
            count += 1
    return "\n".join(lines), count


def fix_model(model: str) -> tuple[str, list[str]]:
    model = map_code(model, to_ascii)  # non-ASCII letters are illegal in identifiers (descriptions keep them)
    model, same_name = fix_same_name_instances(model)
    model, collisions = fix_agent_collisions(model)
    same_name += collisions
    code_only = "".join(STRING_LITERAL.split(model))
    declared = set(DECLARATION.findall(code_only)) | set(INSTANCE.findall(code_only))
    renamed = sorted(declared & SYSML_RESERVED_NAMES)
    for name in renamed:
        new = f"{name}_"
        if re.search(rf"(?<![\w.]){re.escape(new)}(?!\w)", code_only):
            new = f"{name}_name"
        # Only name positions are rewritten, never keyword positions such as `decide node1;`.
        n = re.escape(name)
        pattern = re.compile(
            rf"(?P<pre>(?:\b(?:package|(?:action|part|item|attribute|port|state)(?:[ \t]+def)?|first|then|else|specializes|agent[ \t]*=)|:)[ \t]+)"
            rf"{n}(?!\w)"
            rf"|(?<![\w.]){n}(?=\.\w)"
            rf"|(?<![\w.]){n}(?=::)"
            rf"|(?<=::){n}(?!\w)"
            rf"|(?P<pre2>^[ \t]*){n}(?=[ \t]*:)"
        , re.M)
        model = map_code(model, lambda s, p=pattern, w=new: p.sub(
            lambda m: (m.group("pre") or m.group("pre2") or "") + w if (m.group("pre") or m.group("pre2")) else w, s))
    return model, renamed + same_name


def main() -> int:
    csv.field_size_limit(sys.maxsize)
    with SRC.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        columns = [h.strip('\ufeff"\' ') for h in header]
        col = columns.index("sysml_model")
        rows = list(reader)

    changed_rows, tally, quote_rows = 0, {}, 0
    for row in rows:
        row[col], q = fix_quotes(row[col])
        row[col], b = fix_backslashes(row[col])
        quote_rows += bool(q or b)
        row[col], renamed = fix_model(row[col])
        changed_rows += bool(renamed)
        for name in renamed:
            tally[name] = tally.get(name, 0) + 1

    with DST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"rows={len(rows)} quote_fixed={quote_rows} keyword_fixed={changed_rows} renamed_names={tally}\nwrote {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
