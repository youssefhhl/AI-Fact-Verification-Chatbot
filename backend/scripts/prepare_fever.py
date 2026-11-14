"""Build a ~20k-passage FEVER subset from the official files (offline, run once).

Selection: every Wikipedia sentence cited as evidence by a SUPPORTS/REFUTES claim
in shared_task_dev.jsonl, then sentences cited in train.jsonl (file order) until
TARGET passages are collected. NOT ENOUGH INFO claims have no evidence and are skipped.

Each passage is one evidence sentence, with its page title as context.

Usage (from backend/):  uv run python scripts/prepare_fever.py
"""

import json
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "fever" / "raw"
OUT = RAW.parent / "fever_passages.jsonl"
TARGET = 20_000

# FEVER text is PTB-tokenized; map bracket/colon tokens back to characters.
TOKENS = {
    "-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]",
    "-LCB-": "{", "-RCB-": "}", "-COLON-": ":", "``": '"', "''": '"',
}


def clean(text: str) -> str:
    for token, char in TOKENS.items():
        text = text.replace(token, char)
    return " ".join(text.split())


def select_evidence(claim_files: list[Path], target: int) -> dict[tuple[str, int], list[int]]:
    """Return {(page_id, sentence_id): [claim ids]} for up to `target` sentences."""
    selected: dict[tuple[str, int], list[int]] = {}
    for path in claim_files:
        with path.open() as f:
            for line in f:
                claim = json.loads(line)
                if claim["label"] == "NOT ENOUGH INFO":
                    continue
                keys = {(p, s) for group in claim["evidence"] for _, _, p, s in group}
                new = keys - selected.keys()
                if len(selected) + len(new) > target:
                    return selected
                for key in keys:
                    selected.setdefault(key, []).append(claim["id"])
    return selected


def parse_lines(lines: str) -> dict[int, str]:
    """Parse the wiki `lines` field: 'idx\\tsentence\\t[link...]' per line."""
    sentences: dict[int, str] = {}
    for row in lines.split("\n"):
        parts = row.split("\t")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].strip():
            sentences[int(parts[0])] = parts[1]
    return sentences


def main() -> None:
    selected = select_evidence([RAW / "shared_task_dev.jsonl", RAW / "train.jsonl"], TARGET)
    pages = {page for page, _ in selected}
    print(f"selected {len(selected)} evidence sentences from {len(pages)} pages")

    written = 0
    with zipfile.ZipFile(RAW / "wiki-pages.zip") as zf, OUT.open("w") as out:
        for name in sorted(n for n in zf.namelist() if n.startswith("wiki-pages/wiki-")):
            with zf.open(name) as f:
                for line in f:
                    page = json.loads(line)
                    if page["id"] not in pages:
                        continue
                    for sid, sentence in parse_lines(page["lines"]).items():
                        claim_ids = selected.get((page["id"], sid))
                        if claim_ids is None:
                            continue
                        out.write(json.dumps({
                            "id": f"fever:{page['id']}:{sid}",
                            "page_id": page["id"],
                            "sentence_id": sid,
                            "title": clean(page["id"].replace("_", " ")),
                            "text": clean(sentence),
                            "claim_ids": claim_ids,
                        }) + "\n")
                        written += 1
    print(f"wrote {written} passages to {OUT}")


if __name__ == "__main__":
    main()
