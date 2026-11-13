"""Fix non-ASCII characters in identifiers of the enriched WorFBench SysML v2 models.

Reads  : enriched dataset/worfbench_nemotron_together.csv   (never modified)
Writes : enriched dataset/fixed/worfbench_nemotron_together.csv

Only the `sysml_model` column is changed. Characters such as é, ç, ñ, ä, ģ, ř are not allowed in
SysML v2 identifiers (the lexer fails with "no viable alternative at character ..."), so they are
transliterated to ASCII in the code part of the model (e.g. `LéonDiguetEducation` ->
`LeonDiguetEducation`). Text inside string literals (descriptions) is left untouched, so the
accented spelling is preserved where it is legal and meaningful.

No validation is performed here.
"""
import csv
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "enriched dataset" / "worfbench_nemotron_together.csv"
DST = Path(__file__).resolve().parent / "worfbench_nemotron_together.csv"

STRING_LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"')

# letters that NFKD decomposition does not reduce to a base letter
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


def fix_model(model: str) -> str:
    """Transliterates non-ASCII characters outside string literals."""
    parts, last = [], 0
    for m in STRING_LITERAL.finditer(model):
        parts.append(to_ascii(model[last:m.start()]))
        parts.append(m.group(0))
        last = m.end()
    parts.append(to_ascii(model[last:]))
    return "".join(parts)


def main() -> int:
    csv.field_size_limit(sys.maxsize)
    with SRC.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        columns = [h.strip('\ufeff"\' ') for h in header]
        col = columns.index("sysml_model")
        rows = list(reader)

    changed = 0
    for row in rows:
        fixed = fix_model(row[col])
        changed += fixed != row[col]
        row[col] = fixed

    with DST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"rows={len(rows)} changed={changed}\nwrote {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
