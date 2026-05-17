#!/usr/bin/env python3
"""Prepare SDRL data files from the public Hugging Face sources.

The generated parquet files match the local DAPO/verl-style files expected by
the training scripts:

  data/dapo-math-17k-processed.parquet
  data/math500.parquet
  data/amc23-dapo.parquet
  data/aime24-dapo.parquet
  data/aime25-dapo.parquet
  data/aime24dapo-25-amc23.parquet
"""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


HF_SOURCES = {
    "dapo_math_17k": {
        "repo": "open-r1/DAPO-Math-17k-Processed",
        "path": "all/train-00000-of-00001.parquet",
        "cache_name": "open-r1-DAPO-Math-17k-Processed-all.parquet",
    },
    "math500": {
        "repo": "HuggingFaceH4/MATH-500",
        "path": "test.jsonl",
        "cache_name": "HuggingFaceH4-MATH-500-test.jsonl",
    },
    "amc23": {
        "repo": "math-ai/amc23",
        "path": "test-00000-of-00001.parquet",
        "cache_name": "math-ai-amc23-test.parquet",
    },
    "aime24": {
        "repo": "BytedTsinghua-SIA/AIME-2024",
        "path": "data/aime-2024.parquet",
        "cache_name": "BytedTsinghua-SIA-AIME-2024.parquet",
    },
    "aime25": {
        "repo": "math-ai/aime25",
        "path": "test.jsonl",
        "cache_name": "math-ai-aime25-test.jsonl",
    },
}

ANSWER_STYLE = "rule-lighteval/MATH_v2"
PROMPT_PREFIX = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form Answer: $Answer (without quotes) "
    "where $Answer is the answer to the problem.\n\n"
)
PROMPT_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'

PROMPT_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("content", pa.string()),
            pa.field("role", pa.string()),
        ]
    )
)
REWARD_MODEL_TYPE = pa.struct(
    [
        pa.field("ground_truth", pa.string()),
        pa.field("style", pa.string()),
    ]
)
DAPO_SCHEMA = pa.schema(
    [
        pa.field("prompt", PROMPT_TYPE),
        pa.field("solution", pa.string()),
        pa.field("data_source", pa.string()),
        pa.field("ability", pa.string()),
        pa.field("reward_model", REWARD_MODEL_TYPE),
        pa.field("extra_info", pa.struct([pa.field("index", pa.string())])),
        pa.field("question", pa.string()),
    ]
)
MATH500_SCHEMA = pa.schema(
    [
        pa.field("data_source", pa.string()),
        pa.field("prompt", PROMPT_TYPE),
        pa.field("ability", pa.string()),
        pa.field("reward_model", REWARD_MODEL_TYPE),
        pa.field(
            "extra_info",
            pa.struct(
                [
                    pa.field("raw_problem", pa.string()),
                    pa.field("solution", pa.string()),
                    pa.field("subject", pa.string()),
                    pa.field("unique_id", pa.string()),
                ]
            ),
        ),
    ]
)
EVAL_SCHEMA = pa.schema(
    [
        pa.field("data_source", pa.string()),
        pa.field("prompt", PROMPT_TYPE),
        pa.field("ability", pa.string()),
        pa.field("reward_model", REWARD_MODEL_TYPE),
        pa.field(
            "extra_info",
            pa.struct(
                [
                    pa.field("index", pa.int64()),
                    pa.field("raw_problem", pa.string()),
                    pa.field("split", pa.null()),
                    pa.field("url", pa.string()),
                ]
            ),
        ),
    ]
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data",
        help="Directory where prepared parquet files are written. Defaults to repo_root/data.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "sdrl_hf_data_cache",
        help="Directory for downloaded source files.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download source files again even if they already exist in cache-dir.",
    )
    return parser.parse_args()


def hf_resolve_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def download_sources(cache_dir: Path, force_download: bool) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, source in HF_SOURCES.items():
        target = cache_dir / source["cache_name"]
        paths[key] = target
        if target.exists() and not force_download:
            continue
        url = hf_resolve_url(source["repo"], source["path"])
        print(f"Downloading {source['repo']}::{source['path']} -> {target}")
        urllib.request.urlretrieve(url, target)
    return paths


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    columns = [table[name].to_pylist() for name in table.schema.names]
    return [dict(zip(table.schema.names, row, strict=True)) for row in zip(*columns, strict=True)]


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_rows(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path)
    print(f"Wrote {len(rows):>5} rows -> {path}")


def as_prompt(content: str) -> list[dict[str, str]]:
    return [{"content": content, "role": "user"}]


def wrap_math_prompt(problem: str) -> str:
    return f"{PROMPT_PREFIX}{problem}{PROMPT_SUFFIX}"


def reward_model(answer: Any) -> dict[str, str]:
    return {"ground_truth": normalize_answer(answer), "style": ANSWER_STYLE}


def normalize_answer(answer: Any) -> str:
    if isinstance(answer, float) and answer.is_integer():
        return str(int(answer))
    return str(answer)


def prepare_dapo(source_path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_parquet_rows(source_path):
        rows.append(
            {
                "prompt": row["source_prompt"],
                "solution": row["solution"],
                "data_source": row["data_source"],
                "ability": row["ability"],
                "reward_model": row["reward_model"],
                "extra_info": row["extra_info"],
                "question": row["prompt"],
            }
        )
    return rows


def prepare_math500(source_path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl_rows(source_path):
        rows.append(
            {
                "data_source": "math500",
                "prompt": as_prompt(row["problem"]),
                "ability": "MATH",
                "reward_model": reward_model(row["answer"]),
                "extra_info": {
                    "raw_problem": row["problem"],
                    "solution": row["solution"],
                    "subject": row["subject"],
                    "unique_id": row["unique_id"],
                },
            }
        )
    return rows


def prepare_amc23(source_path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_parquet_rows(source_path):
        rows.append(
            {
                "data_source": "amc23",
                "prompt": as_prompt(wrap_math_prompt(row["question"])),
                "ability": "MATH",
                "reward_model": reward_model(row["answer"]),
                "extra_info": {
                    "index": int(row["id"]),
                    "raw_problem": row["question"],
                    "split": None,
                    "url": row["url"],
                },
            }
        )
    return rows


def prepare_aime24(source_path: Path) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for row in read_parquet_rows(source_path):
        raw_problem = row["extra_info"]["raw_problem"]
        answer = row["reward_model"]["ground_truth"]
        dedup_key = (raw_problem, answer)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        rows.append(
            {
                "data_source": "aime24",
                "prompt": row["prompt"],
                "ability": "MATH",
                "reward_model": row["reward_model"],
                "extra_info": {
                    "index": int(row["extra_info"]["index"]),
                    "raw_problem": raw_problem,
                    "split": row["extra_info"].get("split"),
                    "url": None,
                },
            }
        )
    return rows


def prepare_aime25(source_path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl_rows(source_path):
        rows.append(
            {
                "data_source": "aime25",
                "prompt": as_prompt(wrap_math_prompt(row["problem"])),
                "ability": "MATH",
                "reward_model": reward_model(row["answer"]),
                "extra_info": {
                    "index": int(row["id"]),
                    "raw_problem": None,
                    "split": None,
                    "url": None,
                },
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    sources = download_sources(args.cache_dir, args.force_download)

    dapo_rows = prepare_dapo(sources["dapo_math_17k"])
    math500_rows = prepare_math500(sources["math500"])
    amc23_rows = prepare_amc23(sources["amc23"])
    aime24_rows = prepare_aime24(sources["aime24"])
    aime25_rows = prepare_aime25(sources["aime25"])
    combined_eval_rows = aime24_rows + aime25_rows + amc23_rows

    write_rows(args.output_dir / "dapo-math-17k-processed.parquet", dapo_rows, DAPO_SCHEMA)
    write_rows(args.output_dir / "math500.parquet", math500_rows, MATH500_SCHEMA)
    write_rows(args.output_dir / "amc23-dapo.parquet", amc23_rows, EVAL_SCHEMA)
    write_rows(args.output_dir / "aime24-dapo.parquet", aime24_rows, EVAL_SCHEMA)
    write_rows(args.output_dir / "aime25-dapo.parquet", aime25_rows, EVAL_SCHEMA)
    write_rows(args.output_dir / "aime24dapo-25-amc23.parquet", combined_eval_rows, EVAL_SCHEMA)


if __name__ == "__main__":
    main()
