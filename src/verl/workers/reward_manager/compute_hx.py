"""
compute_hx.py
=============

Purpose
-------
This script computes the step-wise candidate-space entropy described in
Section 3 (Preliminary) of the paper:

    H(X_k) = log |X_k|

where X_k denotes the candidate table region retained after reasoning step k.

The script is designed for the transformed-thinking samples in
`data/ch/sft_think.jsonl` and `data/en/sft_think.jsonl`, where both the
natural-language reasoning trace and the executable Python code follow an
explicit step template:

    Step 1: Path Selection
    Step 2: Column Filtering
    Step 3: Row Filtering
    Step 4: Operation Execution
    Step 5: Answer Generation


What is X_0?
------------
In this implementation, X_0 is defined as the union of all candidate table
files under the same folder as the target file mentioned in the instruction.

Example:
    If the instruction points to
    `盐田预算支出表/盐田预算支出表_一般公共预算支出情况表.csv`,
    then X_0 is built from all csv/xls/xlsx/xlsm files inside
    `.../table_cn/盐田预算支出表/`.

This definition matches the interpretation that before "Path Selection", the
model still needs to choose which table file in the current folder is relevant.


How the script extracts X_t
---------------------------
The extraction logic is a hybrid of structured parsing and concrete execution.

1. Step pairing
   - Parse the `<think> ... </think>` block and split it by `Step n:`.
   - Parse the Python code block and split it by comment headers
     `# Step n: ...`.
   - Pair reasoning text and code by step index.

2. X_0 construction
   - Read the target file path from the instruction.
   - Find the corresponding folder in `table_cn/` or `table/`.
   - Enumerate all candidate tabular files in that folder.
   - Read every candidate table and sum their cell spaces.

3. Step 1: Path Selection
   - Intercept `pd.read_csv(...)`.
   - Record which concrete file is loaded.
   - Bind the loaded DataFrame to its full-table region:
       row_positions = all rows
       col_positions = all columns

4. Step 2: Column Filtering
   - Extract referenced columns from the step text and code.
   - Shrink the current region to the referenced columns while preserving rows.

5. Step 3: Row Filtering
   - Execute the step code.
   - Inspect the resulting DataFrame index.
   - Map the remaining rows back to the original table row positions.

6. Step 4-5: Operation Execution / Answer Generation
   - Track which previously derived table object is being used.
   - If the current expression can still be mapped to a table slice, shrink the
     region further.
   - Otherwise, conservatively inherit the previous candidate region.


Current assumptions
-------------------
This implementation intentionally assumes the structured step template is
present. It is therefore suitable for the transformed-thinking setting, but is
not a fully general provenance engine for arbitrary Python code.

The current implementation works best when:
   - each step is explicitly marked,
   - pandas-style table operations are used,
   - intermediate DataFrame/Series objects remain visible.

It can become less precise when:
   - a step mixes multiple operations heavily,
   - DataFrame -> Series -> scalar transitions are deeply chained,
   - code formatting or indentation inside a step is invalid,
   - later steps operate on heavily transformed derived objects.


Output
------
For each sample, the script prints either:

1. a text summary, or
2. a JSON object containing:
   - the question,
   - all step summaries,
   - candidate cell counts,
   - H(X_t),
   - table path / row range / column range for each inferred region.


Expected X_G input format
-------------------------
This script does not automatically derive the gold support region X_G.
Instead, X_G should be supplied externally in JSON form.

Expected JSON schema:

{
  "gold_regions": [
    {
      "table_path": "ABSOLUTE_OR_RESOLVABLE_TABLE_PATH",
      "row_positions": [15],
      "col_positions": [2, 3]
    }
  ]
}

Semantics:
   - table_path: which original table file the gold support belongs to
   - row_positions: original row indices in that table
   - col_positions: original column indices in that table

Containment rule:
   For every gold region g in X_G, there must exist at least one predicted
   region r in X_t such that:
      - r.table_path == g.table_path
      - g.row_positions ⊆ r.row_positions
      - g.col_positions ⊆ r.col_positions

If the containment rule fails at step t, then:

   H*(X_t) = H(X_0)

Otherwise:

   H*(X_t) = H(X_t)


Reviewer-facing interpretation
------------------------------
This script should be viewed as an executable approximation of the paper's
candidate-space tracking process under the repository's structured step format.

It does not merely search code lines with regex. Instead, it:
   - aligns reasoning steps with code steps,
   - executes step-local code,
   - tracks intermediate table objects,
   - maps them back to original row/column regions,
   - and computes entropy from the resulting candidate-space size.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import math
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["Region", "compute_hx"]


STEP_HEADER_RE = re.compile(r"^\s*#\s*Step\s+(\d+)\s*:\s*(.+?)\s*$")
THINK_STEP_RE = re.compile(
    r"Step\s*(\d+)\s*:\s*(.+?)\n(.*?)(?=\nStep\s*\d+\s*:|\Z)",
    re.DOTALL,
)
CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class Region:
    table_path: str
    row_positions: tuple[int, ...]
    col_positions: tuple[int, ...]

    @property
    def cell_count(self) -> int:
        return len(self.row_positions) * len(self.col_positions)

    def to_summary(self) -> dict[str, Any]:
        return {
            "table_path": self.table_path,
            "row_range": range_summary(self.row_positions),
            "col_range": range_summary(self.col_positions),
            "row_count": len(self.row_positions),
            "col_count": len(self.col_positions),
            "cell_count": self.cell_count,
        }


@dataclass(frozen=True)
class StepPair:
    step_idx: int
    title: str
    think_text: str
    code_text: str


@dataclass(frozen=True)
class StepResult:
    step_idx: int
    title: str
    think_text: str
    code_text: str
    regions: list[Region]
    entropy_hx: float
    effective_entropy_hx: float | None = None
    contains_gold: bool | None = None
    candidate_files: list[str] | None = None

    def to_summary(self) -> dict[str, Any]:
        return {
            "step": self.step_idx,
            "title": self.title,
            "candidate_cells": total_cell_count(self.regions),
            "entropy_hx": self.entropy_hx,
            "effective_entropy_hx": self.effective_entropy_hx,
            "contains_gold": self.contains_gold,
            "candidate_files": self.candidate_files or [],
            "regions": [region.to_summary() for region in self.regions],
        }


class TraceContext:
    def __init__(self, dataset_roots: list[Path]) -> None:
        self.dataset_roots = dataset_roots
        self.tables: dict[str, pd.DataFrame] = {}
        self.dataframe_vars: dict[str, list[Region]] = {}
        self.value_vars: dict[str, list[Region]] = {}
        self.loaded_df_ids: dict[int, str] = {}
        self.last_regions: list[Region] = []
        self.selected_table_paths: list[str] = []
        self._original_read_csv = pd.read_csv

    def resolve_table_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if candidate.exists():
            return candidate.resolve()
        for root in self.dataset_roots:
            joined = root / raw_path
            if joined.exists():
                return joined.resolve()
        raise FileNotFoundError(f"Unable to resolve table path: {raw_path}")

    def tracked_read_csv(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if args:
            raw_path = args[0]
            resolved = self.resolve_table_path(str(raw_path))
            new_args = (str(resolved), *args[1:])
            df = self._original_read_csv(*new_args, **kwargs)
        else:
            raw_path = kwargs.get("filepath_or_buffer")
            resolved = self.resolve_table_path(str(raw_path))
            kwargs = dict(kwargs)
            kwargs["filepath_or_buffer"] = str(resolved)
            df = self._original_read_csv(**kwargs)
        table_key = str(resolved)
        self.tables[table_key] = df
        self.loaded_df_ids[id(df)] = table_key
        return df


def range_summary(values: tuple[int, ...]) -> dict[str, Any]:
    if not values:
        return {"start": None, "end": None, "items": []}
    ordered = sorted(values)
    return {"start": ordered[0], "end": ordered[-1], "items": ordered}


def total_cell_count(regions: list[Region]) -> int:
    return sum(region.cell_count for region in regions)


def compute_h_x(regions: list[Region]) -> float:
    candidate_cells = total_cell_count(regions)
    if candidate_cells <= 0:
        return 0.0
    return math.log(candidate_cells)


def compute_effective_h_x(
    regions: list[Region],
    full_regions: list[Region],
    gold_regions: list[Region] | None = None,
) -> float:
    if not gold_regions:
        return compute_h_x(regions)
    if contains_gold_support(regions, gold_regions):
        return compute_h_x(regions)
    return compute_h_x(full_regions)


def strip_sheet_suffix(table_path: str) -> str:
    return table_path.split("#sheet=", 1)[0]


def normalize_gold_region_input(gold_region: Any) -> list[Region]:
    if gold_region is None:
        return []
    if isinstance(gold_region, Region):
        return [gold_region]
    if isinstance(gold_region, dict):
        if "gold_regions" in gold_region:
            items = gold_region.get("gold_regions", [])
        else:
            items = [gold_region]
        return [parse_region_like(item) for item in items]
    if isinstance(gold_region, (list, tuple)):
        normalized: list[Region] = []
        for item in gold_region:
            if isinstance(item, Region):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(parse_region_like(item))
        return normalized
    raise TypeError("gold_region must be a Region, dict, list of dict/Region, or None")


def extract_raw_table_paths_from_text(text: str) -> list[str]:
    if not text:
        return []
    pattern = re.compile(r"([^\s'\"`]+?\.(?:csv|xlsx|xls|xlsm))", re.IGNORECASE)
    paths = [match.group(1).strip() for match in pattern.finditer(text)]
    return list(dict.fromkeys(paths))


def collect_raw_table_paths_from_spans(
    spans: list[dict[str, Any]],
    code_spans: list[dict[str, Any]],
) -> list[str]:
    raw_paths: list[str] = []
    for span in spans:
        raw_paths.extend(extract_raw_table_paths_from_text(str(span.get("step_text", ""))))
    for code_span in code_spans:
        raw_paths.extend(extract_raw_table_paths_from_text(str(code_span.get("step_code") or "")))
    return list(dict.fromkeys(raw_paths))


def infer_dataset_roots_from_gold_regions(
    gold_regions: list[Region],
    raw_paths: list[str],
) -> list[Path]:
    roots: list[Path] = []
    for region in gold_regions:
        raw_gold_path = strip_sheet_suffix(region.table_path)
        gold_path = Path(raw_gold_path)
        if not gold_path.exists():
            continue
        gold_path = gold_path.resolve()
        roots.append(gold_path.parent)
        if len(gold_path.parents) >= 2:
            roots.append(gold_path.parent.parent)
        for raw_path in raw_paths:
            raw_parts = Path(raw_path).parts
            gold_parts = gold_path.parts
            if len(raw_parts) <= len(gold_parts) and tuple(gold_parts[-len(raw_parts):]) == raw_parts:
                roots.append(Path(*gold_parts[:-len(raw_parts)]))
    return [root for root in dict.fromkeys(roots) if root.exists()]


def build_x0_regions_from_gold_regions(gold_regions: list[Region]) -> list[Region]:
    allowed_suffixes = {".csv", ".xlsx", ".xls", ".xlsm"}
    candidate_regions: list[Region] = []
    visited_files: set[str] = set()
    for gold in gold_regions:
        table_file = Path(strip_sheet_suffix(gold.table_path))
        if not table_file.exists():
            continue
        for child in sorted(table_file.parent.iterdir()):
            if not child.is_file() or child.suffix.lower() not in allowed_suffixes:
                continue
            child_key = str(child.resolve())
            if child_key in visited_files:
                continue
            visited_files.add(child_key)
            candidate_regions.extend(read_tabular_regions(child.resolve()))
    return deduplicate_regions(candidate_regions)


def infer_step_title_from_span(step_idx: int, step_text: str) -> str:
    header_match = re.search(r"Step\s*\d+\s*:\s*(.+)", step_text)
    if header_match:
        return header_match.group(1).strip()
    default_titles = {
        1: "Path Selection",
        2: "Column Filtering",
        3: "Row Filtering",
        4: "Operation Execution",
        5: "Answer Generation",
    }
    return default_titles.get(step_idx, f"Step {step_idx}")


def strip_step_header_from_text(step_text: str) -> str:
    return re.sub(r"^\s*Step\s*\d+\s*:\s*.+?\n?", "", step_text, count=1).strip()


def build_instruction_from_raw_paths(raw_paths: list[str]) -> str:
    return "\n".join(f"文件路径: {raw_path}" for raw_path in raw_paths)


def compute_h_star_from_spans(
    spans: list[dict[str, Any]],
    code_spans: list[dict[str, Any]],
    gold_region: Any,
) -> list[float]:
    """
    Compute step-wise H*(X_t) from already segmented reasoning/code spans.

    Inputs
    ------
    spans:
        A list of step dicts. Each item should contain at least:
            {"step_text": "..."}

    code_spans:
        A list aligned with spans. Each item should contain:
            {"step_code": "..."} or {"step_code": None}

    gold_region:
        Gold support region specification. Accepted forms:
            1. {"table_path": "...", "row_positions": [...], "col_positions": [...]}
            2. {"gold_regions": [ ... ]}
            3. [region_dict_1, region_dict_2, ...]

    Output
    ------
    A list[float] whose length equals len(spans). The i-th value is H*(X_i).
    """
    if len(spans) != len(code_spans):
        raise ValueError("spans and code_spans must have the same length")

    gold_regions = normalize_gold_region_input(gold_region)
    if not gold_regions:
        raise ValueError("gold_region must contain at least one region")

    raw_paths = collect_raw_table_paths_from_spans(spans, code_spans)
    dataset_roots = infer_dataset_roots_from_gold_regions(gold_regions, raw_paths)
    gold_regions = [
        Region(
            table_path=resolve_region_table_path(region.table_path, dataset_roots),
            row_positions=region.row_positions,
            col_positions=region.col_positions,
        )
        for region in gold_regions
    ]

    instruction = build_instruction_from_raw_paths(raw_paths)
    x0_regions = build_x0_regions_from_gold_regions(gold_regions)
    if not x0_regions:
        raise ValueError("Unable to build X_0 from gold_region.table_path")

    ctx = TraceContext(dataset_roots=dataset_roots)
    ctx.last_regions = x0_regions
    env: dict[str, Any] = {"pd": pd}
    original_read_csv = pd.read_csv
    pd.read_csv = ctx.tracked_read_csv
    h_star_values: list[float] = []

    try:
        for step_idx, (span, code_span) in enumerate(zip(spans, code_spans), start=1):
            step_text = str(span.get("step_text", "") or "")
            raw_step_code = code_span.get("step_code")
            normalized_step_code = normalize_code_text(str(raw_step_code or ""))
            step = StepPair(
                step_idx=step_idx,
                title=infer_step_title_from_span(step_idx, step_text),
                think_text=strip_step_header_from_text(step_text),
                code_text=normalized_step_code,
            )

            step_tree = ast.parse(normalized_step_code or "pass")
            if normalized_step_code.strip():
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(normalized_step_code, env, env)

            title_lower = step.title.lower()
            step_regions: list[Region] = []

            if "path selection" in title_lower:
                step_regions = infer_path_selection_regions(step, step_tree, instruction, env, ctx)
            elif "column filtering" in title_lower:
                step_regions = infer_column_step_regions(step, step_tree, instruction, env, ctx)
            elif "row filtering" in title_lower:
                step_regions = infer_assignment_regions(step_tree, env, ctx)
            elif "operation execution" in title_lower:
                step_regions = infer_assignment_regions(step_tree, env, ctx)
            elif "answer generation" in title_lower:
                step_regions = infer_assignment_regions(step_tree, env, ctx)

            if not step_regions:
                step_regions = ctx.last_regions

            ctx.last_regions = step_regions
            h_star_values.append(
                compute_effective_h_x(
                    step_regions,
                    x0_regions,
                    gold_regions=gold_regions,
                )
            )
    finally:
        pd.read_csv = original_read_csv

    return h_star_values


def compute_hx(
    spans: list[dict[str, Any]],
    code_spans: list[dict[str, Any]],
    gold_region: Any,
) -> list[float]:
    """
    Public wrapper used by external methods.

    Input:
        spans, code_spans, gold_region

    Output:
        list[float], where len(output) == len(spans)
        and output[i] is H*(X_i).
    """
    return compute_h_star_from_spans(spans, code_spans, gold_region)


def parse_region_like(obj: dict[str, Any]) -> Region:
    return Region(
        table_path=str(obj["table_path"]),
        row_positions=tuple(int(v) for v in obj["row_positions"]),
        col_positions=tuple(int(v) for v in obj["col_positions"]),
    )


def resolve_region_table_path(raw_table_path: str, dataset_roots: list[Path]) -> str:
    if "#sheet=" in raw_table_path:
        raw_file_path, sheet_name = raw_table_path.split("#sheet=", 1)
        resolved_file_path = resolve_candidate_table_path(raw_file_path, dataset_roots)
        return f"{resolved_file_path}#sheet={sheet_name}"
    return str(resolve_candidate_table_path(raw_table_path, dataset_roots))


def resolve_candidate_table_path(raw_path: str, dataset_roots: list[Path]) -> Path:
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate.resolve()
    for root in dataset_roots:
        joined = root / raw_path
        if joined.exists():
            return joined.resolve()
    return candidate


def load_gold_regions(
    gold_path: Path | None,
    dataset_roots: list[Path] | None = None,
) -> list[Region] | None:
    if gold_path is None:
        return None
    payload = json.loads(gold_path.read_text(encoding="utf-8"))
    dataset_roots = dataset_roots or []
    gold_regions: list[Region] = []
    for item in payload.get("gold_regions", []):
        region = parse_region_like(item)
        gold_regions.append(
            Region(
                table_path=resolve_region_table_path(region.table_path, dataset_roots),
                row_positions=region.row_positions,
                col_positions=region.col_positions,
            )
        )
    return gold_regions


def contains_gold_support(regions: list[Region], gold_regions: list[Region]) -> bool:
    region_index = [
        (
            region.table_path,
            set(region.row_positions),
            set(region.col_positions),
        )
        for region in regions
    ]
    for gold in gold_regions:
        matched = False
        for table_path, rows, cols in region_index:
            if gold.table_path != table_path:
                continue
            if set(gold.row_positions).issubset(rows) and set(gold.col_positions).issubset(cols):
                matched = True
                break
        if not matched:
            return False
    return True


def parse_sample(sample_path: Path, sample_index: int) -> dict[str, Any]:
    with sample_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx == sample_index:
                return json.loads(line)
    raise IndexError(f"Sample index {sample_index} out of range for {sample_path}")


def extract_file_paths_from_instruction(instruction: str) -> list[str]:
    paths: list[str] = []
    for line in instruction.splitlines():
        line = line.strip()
        if line.startswith("文件路径:"):
            paths.append(line.split("文件路径:", 1)[1].strip())
    return paths


def extract_output_sections(output: str) -> tuple[str, str]:
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", output, re.DOTALL)
    code_match = CODE_BLOCK_RE.search(output)
    if not think_match or not code_match:
        raise ValueError("Output does not contain both <think> and ```python blocks")
    return think_match.group(1).strip(), code_match.group(1).strip()


def normalize_code_text(code_text: str) -> str:
    normalized = textwrap.dedent(code_text or "").strip("\n")
    return normalized


def parse_think_steps(think_text: str) -> dict[int, tuple[str, str]]:
    steps: dict[int, tuple[str, str]] = {}
    for match in THINK_STEP_RE.finditer(think_text):
        step_idx = int(match.group(1))
        title = match.group(2).strip()
        body = match.group(3).strip()
        steps[step_idx] = (title, body)
    return steps


def parse_code_steps(code_text: str) -> tuple[str, dict[int, tuple[str, str]]]:
    preamble: list[str] = []
    step_map: dict[int, tuple[str, list[str]]] = {}
    current_step: int | None = None
    current_title = ""

    for line in code_text.splitlines():
        header_match = STEP_HEADER_RE.match(line)
        if header_match:
            current_step = int(header_match.group(1))
            current_title = header_match.group(2).strip()
            step_map[current_step] = (current_title, [])
            continue
        if current_step is None:
            preamble.append(line)
        else:
            step_map[current_step][1].append(line)

    normalized = {
        idx: (title, "\n".join(lines).strip())
        for idx, (title, lines) in step_map.items()
    }
    return "\n".join(preamble).strip(), normalized


def pair_steps(think_text: str, code_text: str) -> tuple[str, list[StepPair]]:
    preamble, code_steps = parse_code_steps(code_text)
    think_steps = parse_think_steps(think_text)
    pairs: list[StepPair] = []
    for step_idx in sorted(set(think_steps) | set(code_steps)):
        think_title, think_body = think_steps.get(step_idx, ("", ""))
        code_title, code_body = code_steps.get(step_idx, ("", ""))
        title = think_title or code_title or f"Step {step_idx}"
        pairs.append(
            StepPair(
                step_idx=step_idx,
                title=title,
                think_text=think_body,
                code_text=normalize_code_text(code_body),
            )
        )
    return normalize_code_text(preamble), pairs


def extract_step_columns(think_text: str, code_text: str) -> list[Any]:
    columns: list[Any] = []
    for pattern in (r"索引\s*([0-9]+)", r"第\s*([0-9]+)\s*列", r"列\s*([0-9]+)"):
        for match in re.findall(pattern, think_text):
            value = int(match)
            if value not in columns:
                columns.append(value)

    try:
        tree = ast.parse(code_text or "")
    except SyntaxError:
        return columns

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            col_values = literal_slice_values(node.slice)
            if col_values:
                for value in col_values:
                    if value not in columns:
                        columns.append(value)
    return columns


def literal_slice_values(node: ast.AST) -> list[Any]:
    if isinstance(node, ast.Constant):
        return [node.value]
    if isinstance(node, ast.List):
        values: list[Any] = []
        for element in node.elts:
            if isinstance(element, ast.Constant):
                values.append(element.value)
        return values
    if isinstance(node, ast.Tuple):
        values = []
        for element in node.elts:
            if isinstance(element, ast.Constant):
                values.append(element.value)
        return values
    return []


def safe_eval(node: ast.AST, env: dict[str, Any]) -> Any:
    expression = ast.Expression(body=node)
    return eval(compile(expression, "<ast>", "eval"), env, env)


def map_columns_to_positions(df: pd.DataFrame, columns: list[Any]) -> tuple[int, ...]:
    positions: list[int] = []
    for col in columns:
        if col in df.columns:
            loc = df.columns.get_loc(col)
            if isinstance(loc, slice):
                positions.extend(range(loc.start or 0, loc.stop or 0))
            elif isinstance(loc, list):
                positions.extend(int(v) for v in loc)
            elif hasattr(loc, "tolist"):
                positions.extend(int(v) for v in loc.tolist())
            else:
                positions.append(int(loc))
    return tuple(sorted(dict.fromkeys(positions)))


def map_index_values_to_positions(index: pd.Index, values: Any) -> tuple[int, ...]:
    if isinstance(values, pd.Index):
        labels = list(values)
    elif isinstance(values, (list, tuple)):
        labels = list(values)
    else:
        labels = [values]
    positions: list[int] = []
    for label in labels:
        loc = index.get_loc(label)
        if isinstance(loc, slice):
            positions.extend(range(loc.start or 0, loc.stop or 0))
        elif isinstance(loc, list):
            positions.extend(int(v) for v in loc)
        elif hasattr(loc, "tolist"):
            positions.extend(int(v) for v in loc.tolist())
        else:
            positions.append(int(loc))
    return tuple(sorted(dict.fromkeys(positions)))


def full_region_for_table(table_path: str, df: pd.DataFrame) -> Region:
    return Region(
        table_path=table_path,
        row_positions=tuple(range(len(df.index))),
        col_positions=tuple(range(len(df.columns))),
    )


def read_tabular_regions(file_path: Path) -> list[Region]:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file_path, header=None)
        return [full_region_for_table(str(file_path), df)]

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        excel_data = pd.read_excel(file_path, sheet_name=None, header=None)
        regions: list[Region] = []
        for sheet_name, df in excel_data.items():
            regions.append(full_region_for_table(f"{file_path}#sheet={sheet_name}", df))
        return regions

    return []


def regions_for_resolved_path(file_path: Path) -> list[Region]:
    return deduplicate_regions(read_tabular_regions(file_path))


def project_runtime_object_to_region(
    source_region: Region,
    runtime_obj: Any,
    table_df: pd.DataFrame,
) -> Region:
    if isinstance(runtime_obj, pd.DataFrame):
        row_positions = map_index_values_to_positions(table_df.index, runtime_obj.index)
        col_positions = map_columns_to_positions(table_df, list(runtime_obj.columns))
        return Region(source_region.table_path, row_positions, col_positions)
    if isinstance(runtime_obj, pd.Series):
        row_positions = map_index_values_to_positions(table_df.index, runtime_obj.index)
        return Region(source_region.table_path, row_positions, source_region.col_positions)
    return source_region


def infer_regions_from_expr(node: ast.AST, env: dict[str, Any], ctx: TraceContext) -> list[Region]:
    if isinstance(node, ast.Name):
        if node.id in ctx.value_vars:
            return ctx.value_vars[node.id]
        if node.id in ctx.dataframe_vars:
            return ctx.dataframe_vars[node.id]
        return []

    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id in ctx.dataframe_vars:
            source_name = node.value.id
            source_regions = ctx.dataframe_vars[source_name]
            runtime_obj = safe_eval(node, env)
            table_path = source_regions[0].table_path
            table_df = ctx.tables[table_path]
            if isinstance(runtime_obj, pd.DataFrame):
                return [project_runtime_object_to_region(source_regions[0], runtime_obj, table_df)]
            if isinstance(runtime_obj, pd.Series):
                col_positions = source_regions[0].col_positions
                col_values = literal_slice_values(node.slice)
                if col_values:
                    col_positions = map_columns_to_positions(table_df, col_values)
                row_positions = map_index_values_to_positions(table_df.index, runtime_obj.index)
                return [Region(table_path, row_positions, col_positions)]
            return source_regions

        if (
            isinstance(node.value, ast.Attribute)
            and node.value.attr in {"iloc", "loc"}
        ):
            base_regions = infer_regions_from_expr(node.value.value, env, ctx)
            if not base_regions:
                return []
            base_region = base_regions[0]
            base_runtime = safe_eval(node.value.value, env)
            table_df = ctx.tables[base_region.table_path]
            if isinstance(base_runtime, pd.Series):
                selected = safe_eval(node, env)
                if isinstance(selected, pd.Series):
                    row_positions = map_index_values_to_positions(table_df.index, selected.index)
                else:
                    row_key = safe_eval(node.slice, env)
                    if isinstance(row_key, int):
                        row_positions = map_index_values_to_positions(
                            table_df.index, [base_runtime.index[row_key]]
                        )
                    else:
                        row_positions = base_region.row_positions
                return [Region(base_region.table_path, row_positions, base_region.col_positions)]
            if isinstance(base_runtime, pd.DataFrame):
                selected = safe_eval(node, env)
                if isinstance(selected, pd.DataFrame):
                    return [project_runtime_object_to_region(base_region, selected, table_df)]
                if isinstance(selected, pd.Series):
                    return [project_runtime_object_to_region(base_region, selected, table_df)]
                return base_regions

        return infer_regions_from_expr(node.value, env, ctx)

    if isinstance(node, ast.Attribute):
        return infer_regions_from_expr(node.value, env, ctx)

    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "read_csv"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pd"
        ):
            runtime_obj = safe_eval(node, env)
            table_path = ctx.loaded_df_ids.get(id(runtime_obj))
            if not table_path:
                return []
            return [full_region_for_table(table_path, runtime_obj)]

        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"sum", "mean", "max", "min", "median", "count", "idxmax", "idxmin"}
        ):
            return infer_regions_from_expr(node.func.value, env, ctx)

        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"copy", "dropna", "sort_values", "head", "tail", "reset_index"}
        ):
            base_regions = infer_regions_from_expr(node.func.value, env, ctx)
            if not base_regions:
                return []
            runtime_obj = safe_eval(node, env)
            base_region = base_regions[0]
            table_df = ctx.tables[base_region.table_path]
            if isinstance(runtime_obj, (pd.DataFrame, pd.Series)):
                return [project_runtime_object_to_region(base_region, runtime_obj, table_df)]
            return base_regions

        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "isin"
        ):
            return infer_regions_from_expr(node.func.value, env, ctx)

        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "astype"
        ):
            return infer_regions_from_expr(node.func.value, env, ctx)

        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pd"
            and node.func.attr == "to_numeric"
            and node.args
        ):
            return infer_regions_from_expr(node.args[0], env, ctx)

        regions: list[Region] = []
        for arg in node.args:
            regions.extend(infer_regions_from_expr(arg, env, ctx))
        for keyword in node.keywords:
            regions.extend(infer_regions_from_expr(keyword.value, env, ctx))
        return deduplicate_regions(regions)

    if isinstance(node, ast.BinOp):
        return deduplicate_regions(
            infer_regions_from_expr(node.left, env, ctx)
            + infer_regions_from_expr(node.right, env, ctx)
        )

    if isinstance(node, ast.Compare):
        regions = infer_regions_from_expr(node.left, env, ctx)
        for comparator in node.comparators:
            regions.extend(infer_regions_from_expr(comparator, env, ctx))
        return deduplicate_regions(regions)

    if isinstance(node, ast.BoolOp):
        regions: list[Region] = []
        for value in node.values:
            regions.extend(infer_regions_from_expr(value, env, ctx))
        return deduplicate_regions(regions)

    return []


def deduplicate_regions(regions: list[Region]) -> list[Region]:
    return list(dict.fromkeys(regions))


def infer_assignment_regions(
    step_tree: ast.Module,
    env: dict[str, Any],
    ctx: TraceContext,
) -> list[Region]:
    assigned_regions: list[Region] = []
    for node in step_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target_names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if not target_names:
            continue
        value_regions = infer_regions_from_expr(node.value, env, ctx)
        for target_name in target_names:
            runtime_obj = env.get(target_name)
            if isinstance(runtime_obj, pd.DataFrame):
                if value_regions:
                    ctx.dataframe_vars[target_name] = value_regions
                    assigned_regions = value_regions
            elif value_regions:
                ctx.value_vars[target_name] = value_regions
                assigned_regions = value_regions
    return assigned_regions


def infer_selected_paths_from_env(
    step_tree: ast.Module,
    env: dict[str, Any],
    ctx: TraceContext,
) -> list[str]:
    selected_paths: list[str] = []
    for node in step_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            runtime_obj = env.get(target.id)
            if isinstance(runtime_obj, str):
                try:
                    resolved = ctx.resolve_table_path(runtime_obj)
                except Exception:
                    continue
                selected_paths.append(str(resolved))
    return list(dict.fromkeys(selected_paths))


def infer_path_selection_regions(
    step: StepPair,
    step_tree: ast.Module,
    instruction: str,
    env: dict[str, Any],
    ctx: TraceContext,
) -> list[Region]:
    step_regions = infer_assignment_regions(step_tree, env, ctx)
    if step_regions:
        return deduplicate_regions(step_regions)

    selected_paths = infer_selected_paths_from_env(step_tree, env, ctx)
    if not selected_paths:
        resolved_instruction_paths = resolve_instruction_table_paths(instruction, ctx.dataset_roots)
        selected_paths = [str(path) for path in resolved_instruction_paths]

    ctx.selected_table_paths = selected_paths
    step_regions = []
    for selected_path in selected_paths:
        step_regions.extend(regions_for_resolved_path(Path(selected_path)))
    return deduplicate_regions(step_regions)


def infer_column_filter_regions(
    step: StepPair,
    env: dict[str, Any],
    ctx: TraceContext,
) -> list[Region]:
    columns = extract_step_columns(step.think_text, step.code_text)
    if not columns:
        return ctx.last_regions

    referenced_vars = sorted({
        node.id
        for node in ast.walk(ast.parse(step.code_text or "pass"))
        if isinstance(node, ast.Name) and node.id in ctx.dataframe_vars
    })
    if not referenced_vars:
        referenced_vars = list(ctx.dataframe_vars.keys())

    regions: list[Region] = []
    for var_name in referenced_vars:
        current_regions = ctx.dataframe_vars.get(var_name, [])
        if not current_regions:
            continue
        narrowed_for_var: list[Region] = []
        for region in current_regions:
            table_df = ctx.tables[region.table_path]
            col_positions = map_columns_to_positions(table_df, columns)
            if not col_positions:
                continue
            narrowed_for_var.append(
                Region(region.table_path, region.row_positions, col_positions)
            )
        if narrowed_for_var:
            ctx.dataframe_vars[var_name] = narrowed_for_var
            regions.extend(narrowed_for_var)
    return deduplicate_regions(regions)


def infer_column_step_regions(
    step: StepPair,
    step_tree: ast.Module,
    instruction: str,
    env: dict[str, Any],
    ctx: TraceContext,
) -> list[Region]:
    assigned_regions = infer_assignment_regions(step_tree, env, ctx)

    if not ctx.dataframe_vars and ctx.selected_table_paths:
        for selected_path in ctx.selected_table_paths:
            regions = regions_for_resolved_path(Path(selected_path))
            if not regions:
                continue
            for var_name, runtime_obj in env.items():
                if isinstance(runtime_obj, pd.DataFrame):
                    table_key = ctx.loaded_df_ids.get(id(runtime_obj))
                    if table_key == selected_path:
                        ctx.dataframe_vars[var_name] = regions

    narrowed_regions = infer_column_filter_regions(step, env, ctx)
    if narrowed_regions:
        return narrowed_regions
    if assigned_regions:
        return assigned_regions

    resolved_instruction_paths = resolve_instruction_table_paths(instruction, ctx.dataset_roots)
    fallback_regions: list[Region] = []
    for path in resolved_instruction_paths:
        fallback_regions.extend(regions_for_resolved_path(path))
    return deduplicate_regions(fallback_regions)


def build_llm_judge_prompt(
    instruction: str,
    step: StepPair,
    current_regions: list[Region],
) -> str:
    region_lines = []
    for region in current_regions:
        region_lines.append(
            f"- table={region.table_path}, rows={list(region.row_positions)}, cols={list(region.col_positions)}"
        )
    region_block = "\n".join(region_lines) if region_lines else "- none"
    return (
        "You are helping recover a candidate table region X_t for a structured TableQA step.\n"
        "Return JSON only with keys: mode, reason.\n"
        "mode must be one of: keep_previous, use_selected_table, unresolved.\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"Step title: {step.title}\n"
        f"Step reasoning:\n{step.think_text}\n\n"
        f"Step code:\n{step.code_text}\n\n"
        f"Current inferred regions:\n{region_block}\n"
    )


def maybe_llm_judge_regions(
    instruction: str,
    step: StepPair,
    current_regions: list[Region],
    ctx: TraceContext,
) -> list[Region]:
    if current_regions:
        return current_regions

    if not os.environ.get("OPENAI_API_KEY"):
        return current_regions

    # Optional hook only. We keep the implementation conservative so that the
    # script remains fully usable without network or credentials.
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return current_regions

    base_url = os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("OPENAI_MODEL")
    if not model:
        return current_regions

    prompt = build_llm_judge_prompt(instruction, step, ctx.last_regions)
    try:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return current_regions
        payload = json.loads(match.group(0))
        mode = payload.get("mode")
        if mode == "keep_previous":
            return ctx.last_regions
        if mode == "use_selected_table" and ctx.selected_table_paths:
            judged_regions: list[Region] = []
            for selected_path in ctx.selected_table_paths:
                judged_regions.extend(regions_for_resolved_path(Path(selected_path)))
            return deduplicate_regions(judged_regions)
    except Exception:
        return current_regions

    return current_regions


def execute_trace(sample_path: Path, sample_index: int) -> tuple[dict[str, Any], list[StepResult]]:
    sample = parse_sample(sample_path, sample_index)
    instruction = sample["instruction"]
    output = sample["output"]
    think_text, code_text = extract_output_sections(output)
    preamble, step_pairs = pair_steps(think_text, code_text)

    dataset_roots = infer_dataset_roots(sample_path)
    x0_result = build_x0_result(instruction, dataset_roots)
    ctx = TraceContext(dataset_roots=dataset_roots)

    env: dict[str, Any] = {"pd": pd}
    original_read_csv = pd.read_csv
    pd.read_csv = ctx.tracked_read_csv
    try:
        if preamble:
            exec(preamble, env, env)

        results: list[StepResult] = [x0_result]
        for step in step_pairs:
            normalized_step_code = normalize_code_text(step.code_text)
            step_tree = ast.parse(normalized_step_code or "pass")
            if normalized_step_code.strip():
                exec(normalized_step_code, env, env)

            title_lower = step.title.lower()
            step_regions: list[Region] = []

            if "path selection" in title_lower:
                step_regions = infer_path_selection_regions(step, step_tree, instruction, env, ctx)
            elif "column filtering" in title_lower:
                step_regions = infer_column_step_regions(step, step_tree, instruction, env, ctx)
            elif "row filtering" in title_lower:
                step_regions = infer_assignment_regions(step_tree, env, ctx)
            elif "operation execution" in title_lower:
                step_regions = infer_assignment_regions(step_tree, env, ctx)
            elif "answer generation" in title_lower:
                step_regions = infer_assignment_regions(step_tree, env, ctx)

            if not step_regions:
                step_regions = ctx.last_regions

            step_regions = maybe_llm_judge_regions(instruction, step, step_regions, ctx)

            ctx.last_regions = step_regions
            results.append(
                StepResult(
                    step_idx=step.step_idx,
                    title=step.title,
                    think_text=step.think_text,
                    code_text=normalized_step_code,
                    regions=step_regions,
                    entropy_hx=compute_h_x(step_regions),
                )
            )

        return sample, results
    finally:
        pd.read_csv = original_read_csv


def attach_effective_entropy(
    step_results: list[StepResult],
    gold_regions: list[Region] | None,
) -> list[StepResult]:
    if not step_results:
        return step_results

    x0_regions = step_results[0].regions
    enriched: list[StepResult] = []
    for result in step_results:
        contains_gold = None
        effective_entropy_hx = None
        if gold_regions is not None:
            contains_gold = contains_gold_support(result.regions, gold_regions)
            effective_entropy_hx = compute_effective_h_x(
                result.regions,
                x0_regions,
                gold_regions=gold_regions,
            )
        enriched.append(
            StepResult(
                step_idx=result.step_idx,
                title=result.title,
                think_text=result.think_text,
                code_text=result.code_text,
                regions=result.regions,
                entropy_hx=result.entropy_hx,
                effective_entropy_hx=effective_entropy_hx,
                contains_gold=contains_gold,
                candidate_files=result.candidate_files,
            )
        )
    return enriched


def infer_dataset_roots(sample_path: Path) -> list[Path]:
    path_str = str(sample_path).replace("\\", "/")
    roots: list[Path] = []
    if "/data/ch/" in path_str:
        roots.append(sample_path.parents[1] / "table_cn")
    if "/data/en/" in path_str:
        roots.append(sample_path.parents[1] / "table")
    project_root = sample_path.parents[2]
    roots.append(project_root / "data" / "ch" / "table_cn")
    roots.append(project_root / "data" / "en" / "table")
    return [root for root in roots if root.exists()]


def resolve_instruction_table_paths(
    instruction: str,
    dataset_roots: list[Path],
) -> list[Path]:
    resolved: list[Path] = []
    raw_paths = extract_file_paths_from_instruction(instruction)
    for raw_path in raw_paths:
        candidate = Path(raw_path)
        if candidate.exists():
            resolved.append(candidate.resolve())
            continue
        for root in dataset_roots:
            joined = root / raw_path
            if joined.exists():
                resolved.append(joined.resolve())
                break
    return resolved


def collect_candidate_files(instruction: str, dataset_roots: list[Path]) -> list[Path]:
    resolved_paths = resolve_instruction_table_paths(instruction, dataset_roots)
    candidate_files: list[Path] = []
    allowed_suffixes = {".csv", ".xlsx", ".xls", ".xlsm"}
    for resolved_path in resolved_paths:
        folder = resolved_path.parent
        for child in sorted(folder.iterdir()):
            if child.is_file() and child.suffix.lower() in allowed_suffixes:
                candidate_files.append(child.resolve())
    if candidate_files:
        return list(dict.fromkeys(candidate_files))
    return resolved_paths


def build_x0_result(instruction: str, dataset_roots: list[Path]) -> StepResult:
    candidate_files = collect_candidate_files(instruction, dataset_roots)
    candidate_regions: list[Region] = []
    for candidate_file in candidate_files:
        candidate_regions.extend(read_tabular_regions(candidate_file))
    return StepResult(
        step_idx=0,
        title="Initial Candidate Tables",
        think_text="Before path selection, X_0 contains every candidate table file in the current folder.",
        code_text="",
        regions=deduplicate_regions(candidate_regions),
        entropy_hx=compute_h_x(candidate_regions),
        candidate_files=[path.name for path in candidate_files],
    )


def extract_question_from_instruction(instruction: str) -> str:
    patterns = [
        r"输入问题[:：]\s*(.+)",
        r"Question[:：]\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, instruction)
        if match:
            return match.group(1).strip()
    lines = [line.strip() for line in instruction.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def build_input_dump(
    sample: dict[str, Any],
    sample_path: Path,
    sample_index: int,
) -> dict[str, Any]:
    instruction = sample["instruction"]
    output = sample["output"]
    think_text, code_text = extract_output_sections(output)
    preamble, step_pairs = pair_steps(think_text, code_text)
    dataset_roots = infer_dataset_roots(sample_path)
    resolved_paths = resolve_instruction_table_paths(instruction, dataset_roots)
    candidate_files = collect_candidate_files(instruction, dataset_roots)
    return {
        "sample_path": str(sample_path),
        "sample_index": sample_index,
        "question": extract_question_from_instruction(instruction),
        "instruction": instruction,
        "instruction_file_paths": extract_file_paths_from_instruction(instruction),
        "resolved_instruction_tables": [str(path) for path in resolved_paths],
        "x0_candidate_files": [str(path) for path in candidate_files],
        "raw_output": output,
        "think_text": think_text,
        "code_text": code_text,
        "preamble_code": preamble,
        "step_pairs": [
            {
                "step": step.step_idx,
                "title": step.title,
                "think_text": step.think_text,
                "code_text": step.code_text,
            }
            for step in step_pairs
        ],
    }


def print_summary(sample: dict[str, Any], step_results: list[StepResult]) -> None:
    instruction = sample["instruction"]
    question = instruction.split("输入问题：")[-1].strip()
    file_paths = extract_file_paths_from_instruction(instruction)
    print("Question:", question)
    print("Instruction file paths:", file_paths)
    print()
    for result in step_results:
        print(f"Step {result.step_idx}: {result.title}")
        print(f"  H(X_t) = {result.entropy_hx:.6f}")
        if result.effective_entropy_hx is not None:
            print(f"  H*(X_t) = {result.effective_entropy_hx:.6f}")
            print(f"  X_G subset of X_t: {result.contains_gold}")
        print(f"  |X_t| = {total_cell_count(result.regions)}")
        if result.candidate_files:
            print(f"  Candidate files: {result.candidate_files}")
        for region in result.regions:
            summary = region.to_summary()
            print(f"  Table: {summary['table_path']}")
            print(
                "    Rows:",
                f"{summary['row_range']['start']} -> {summary['row_range']['end']}",
                f"(count={summary['row_count']})",
            )
            print(
                "    Cols:",
                f"{summary['col_range']['start']} -> {summary['col_range']['end']}",
                f"(count={summary['col_count']})",
            )
            print(f"    Cells: {summary['cell_count']}")
        print()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute H(X_t)=log|X_t| for transformed-thinking TableQA traces."
    )
    parser.add_argument(
        "--sample-path",
        type=Path,
        default=Path(r"D:\PycharmProjects\ReasonTabQA\UR-TabQA\data\ch\sft_think.jsonl"),
        help="Path to the transformed-thinking jsonl file.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="0-based sample index inside the jsonl file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a text summary.",
    )
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=None,
        help="Optional JSON file that supplies externally prepared gold support regions X_G.",
    )
    parser.add_argument(
        "--dump-input",
        action="store_true",
        help="Print the parsed input content needed to prepare X_G and verify step splitting.",
    )
    return parser


def build_xg_schema_description() -> dict[str, Any]:
    return {
        "description": "Externally provided gold support region X_G used for effective entropy H*(X_t).",
        "json_format": {
            "gold_regions": [
                {
                    "table_path": "ABSOLUTE_OR_RESOLVABLE_TABLE_PATH",
                    "row_positions": [15],
                    "col_positions": [2, 3],
                }
            ]
        },
        "semantics": {
            "table_path": "Original table path for the gold support region.",
            "row_positions": "Original row indices in the source table.",
            "col_positions": "Original column indices in the source table.",
        },
        "containment_rule": (
            "For every gold region, there must exist at least one predicted region with "
            "the same table_path and a superset of the gold rows and columns. "
            "Otherwise H*(X_t)=H(X_0)."
        ),
    }


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    sample, step_results = execute_trace(args.sample_path, args.sample_index)
    dataset_roots = infer_dataset_roots(args.sample_path)
    gold_regions = load_gold_regions(args.gold_path, dataset_roots)
    step_results = attach_effective_entropy(step_results, gold_regions)
    input_dump = build_input_dump(sample, args.sample_path, args.sample_index)
    if args.json:
        payload = {
            "question": extract_question_from_instruction(sample["instruction"]),
            "input_dump": input_dump,
            "xg_input_schema": build_xg_schema_description(),
            "gold_regions": [
                {
                    "table_path": region.table_path,
                    "row_positions": list(region.row_positions),
                    "col_positions": list(region.col_positions),
                }
                for region in (gold_regions or [])
            ],
            "steps": [result.to_summary() for result in step_results],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if args.dump_input:
        print(json.dumps(input_dump, ensure_ascii=False, indent=2))
        print()
    print_summary(sample, step_results)


if __name__ == "__main__":
    main()
