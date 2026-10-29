import json
from pathlib import Path

from chunkyard import fix, from_records, lint, load_chunks
from chunkyard.cli import main

EX = Path(__file__).parent.parent / "examples" / "chunks.jsonl"


def codes(r):
    return {f.code for f in r.findings}


def by_code(r, code):
    return [f.id for f in r.findings if f.code == code]


def test_example_corpus():
    r = lint(load_chunks(EX))
    assert codes(r) == {"C001", "C002", "C004", "C005", "C006", "C007", "C008", "C010", "C011"}
    assert by_code(r, "C004") == ["c6"] and by_code(r, "C005") == ["c7"]
    assert by_code(r, "C001") == ["c8"] and by_code(r, "C010") == ["c11"]
    assert r.stats["documents"] == 8 and r.stats["boilerplate_token_pct"] > 20
    assert not r.ok


def test_boilerplate_lines_are_learned_from_the_corpus():
    r = lint(load_chunks(EX))
    assert any("all rights reserved" in line for line in r.boilerplate)
    assert any("home docs help center" in line for line in r.boilerplate)


def test_fix_removes_duplicates_and_boilerplate():
    r = lint(load_chunks(EX))
    rows = fix(r)
    texts = [x["text"] for x in rows]
    assert len(rows) == len(r.chunks) - 3
    assert not any("All rights reserved" in t for t in texts)
    assert rows[0]["metadata"]["source"] == "billing.md"
    again = lint(from_records(rows))
    assert not {"C001", "C004", "C005", "C006"} & codes(again)


def test_sizes_and_cut_sentences():
    long_text = " ".join(f"word{i}" for i in range(400))
    r = lint(from_records([{"id": "a", "text": long_text, "source": "x"}]), max_tokens=300)
    assert {"C003", "C009"} <= codes(r)


def test_keys_detection_and_plain_strings(tmp_path):
    rows = [
        {"chunk_id": 7, "page_content": "LangChain style document with enough words to pass the minimum length check easily.", "metadata": {"url": "https://x"}}
    ]
    c = from_records(rows)[0]
    assert (c.id, c.doc, c.text_key) == (7, "https://x", "page_content")
    f = tmp_path / "c.json"
    f.write_text(json.dumps(["plain string chunk"]))
    assert load_chunks(f)[0].text == "plain string chunk"
    csvf = tmp_path / "c.csv"
    csvf.write_text("id,content,source\n1,hello world,doc.md\n")
    assert load_chunks(csvf)[0].doc == "doc.md"


def test_clean_corpus_passes():
    rows = [
        {
            "id": i,
            "text": f"Paragraph {i} explains a different topic in enough detail: topic number {i} covers item {i * 7} and case {i * 13}.",
            "source": f"d{i}",
        }
        for i in range(5)
    ]
    r = lint(from_records(rows), min_tokens=5)
    assert r.findings == [] and r.ok


def test_report_text(tmp_path, capsys):
    out = tmp_path / "clean.jsonl"
    assert main([str(EX), "--fix", str(out)]) == 1
    text = capsys.readouterr().out
    assert "near-duplicates" in text and "wrote" in text
    assert main([str(out), "--min-tokens", "5"]) == 0
    capsys.readouterr()
    assert main([str(EX), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["counts"]["C004"] == 1
