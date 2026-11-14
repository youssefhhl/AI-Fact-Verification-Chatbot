import json
from pathlib import Path

from scripts.prepare_fever import clean, parse_lines, select_evidence


def test_clean_restores_brackets() -> None:
    assert clean("Gladiator -LRB-2000 film-RRB-  ``x''") == 'Gladiator (2000 film) "x"'


def test_parse_lines_skips_empty_sentences() -> None:
    lines = "0\tFirst sentence .\tLink\n1\t\n2\tThird ."
    assert parse_lines(lines) == {0: "First sentence .", 2: "Third ."}


def test_select_evidence_skips_nei_and_respects_target(tmp_path: Path) -> None:
    claims = [
        {"id": 1, "label": "SUPPORTS", "evidence": [[[0, 0, "A", 0]], [[0, 0, "A", 1]]]},
        {"id": 2, "label": "NOT ENOUGH INFO", "evidence": [[[0, 0, None, None]]]},
        {"id": 3, "label": "REFUTES", "evidence": [[[0, 0, "A", 0]]]},
        {"id": 4, "label": "SUPPORTS", "evidence": [[[0, 0, "B", 0]]]},
    ]
    path = tmp_path / "claims.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in claims))

    selected = select_evidence([path], target=2)

    assert selected == {("A", 0): [1, 3], ("A", 1): [1]}
