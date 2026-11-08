#!/usr/bin/env python3
"""严格 loss 验收工具（002-P1 v2 §A / 006-P1 v2）。

只用标准库。判据来源分开标注：
  阈值 FP32 max_abs<=1e-5 / BF16 max_rel<=1e-2 —— criterion_source=user_pdf_and_a100_log
  相对误差分母、零参考处理、<= 边界 —— comparison_policy_source=program_convention

参考登记约定：registry 里的相对路径以 registry 文件所在目录解析；单 case 文件必须
能由 raw 全日志的标题精确匹配重新切出并与冻结文件逐字节一致，否则拒绝。

退出码：0 通过；2 精度不达标；3 输入/格式/元数据/参考登记不合法。
任何拒绝条件命中都不得打印 PASS；诊断模式只输出 DIAGNOSTIC_*。
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys

# A-R3：数字必须在真实分隔符处结束（拒绝 5.0e+garbage 这类被截断的写法）
NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
STEP_RE = re.compile(
    r"step\s+(?P<step>\d+)\s*/\s*(?P<total>\d+)\s*\|\s*train loss\s+"
    r"(?P<loss>" + NUM + r"|[-+]?(?:nan|inf|infinity))"
    r"(?=[\s|,;]|$)",
    re.IGNORECASE,
)
RANK_RE = re.compile(r"(?:^|\s)Rank\s+(\d+)\s*:")
FP32_ATOL = 1e-5
BF16_RTOL = 1e-2


class Reject(Exception):
    """输入或元数据不合法（退出码 3）。"""


class PrecisionFail(Exception):
    """精度不达标（退出码 2）。"""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
CASE_TITLE_PREFIX = b"Running:"


def norm_path(path):
    """Normalize a path for identity comparison (Windows-safe)."""
    return os.path.normcase(os.path.realpath(path))


def is_separator(raw_line):
    """True for a decoration line made only of '=' characters."""
    stripped = raw_line.strip()
    return bool(stripped) and set(stripped) == {ord("=")}


def split_case_segment(raw_path, title):
    """Return the raw bytes of the single case segment named by `title`.

    The segment starts at the line that equals `title` exactly (so the
    `gpt2_1_bfloat16` case cannot match `gpt2_1 (fp32)`), includes the title
    decoration separator that follows the title, and ends right before the
    separator line that introduces the next case (a separator line followed by
    a `Running:` line) or at end of file. Line positions are discovered from
    the content, never hardcoded.
    """
    with open(raw_path, "rb") as f:
        raw_lines = f.read().splitlines(keepends=True)
    wanted = title.encode("utf-8")
    hits = [i for i, line in enumerate(raw_lines) if line.rstrip(b"\r\n") == wanted]
    if not hits:
        raise Reject("case title {!r} not found in raw log {}".format(title, raw_path))
    if len(hits) > 1:
        raise Reject(
            "case title {!r} appears {} times in {} (ambiguous, refusing)".format(
                title, len(hits), raw_path
            )
        )
    start = hits[0]
    if start + 1 >= len(raw_lines) or not is_separator(raw_lines[start + 1]):
        raise Reject(
            "case title {!r} in {} is not followed by a separator line".format(
                title, raw_path
            )
        )
    end = len(raw_lines)
    for index in range(start + 2, len(raw_lines)):
        if not is_separator(raw_lines[index]):
            continue
        if index + 1 >= len(raw_lines):
            end = index
            break
        if raw_lines[index + 1].lstrip().startswith(CASE_TITLE_PREFIX):
            end = index
            break
    segment = b"".join(raw_lines[start:end])
    if not segment:
        raise Reject("empty case segment for {!r} in {}".format(title, raw_path))
    return segment


def verify_derived_provenance(raw_path, derived_path, title):
    """Prove the single-case file really is the frozen segment of the raw log."""
    segment = split_case_segment(raw_path, title)
    with open(derived_path, "rb") as f:
        derived = f.read()
    if segment != derived:
        raise Reject(
            "derived reference does not match the case segment of the raw log: "
            "raw_segment_sha256={} derived_sha256={} title={!r}".format(
                hashlib.sha256(segment).hexdigest(),
                hashlib.sha256(derived).hexdigest(),
                title,
            )
        )


def parse_losses(path, expected_steps, step_base, rank):
    if not os.path.isfile(path):
        raise Reject("log not found: {}".format(path))
    steps = {}
    ranks_seen = set()
    with open(path, "r", encoding="utf-8", errors="replace") as _pf:
        for _line in _pf:
            _m = RANK_RE.search(_line)
            if _m:
                ranks_seen.add(int(_m.group(1)))
    if rank is None and len(ranks_seen) > 1:
        raise Reject(
            "ambiguous multi-rank log {} (ranks={}); pass --rank".format(
                path, sorted(ranks_seen)
            )
        )
    untagged_loss = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            m_rank = RANK_RE.search(line)
            if m_rank:
                ranks_seen.add(int(m_rank.group(1)))
                if rank is not None and int(m_rank.group(1)) != rank:
                    continue
            m = STEP_RE.search(line)
            if m and m_rank is None:
                untagged_loss.append(lineno)
            if not m:
                if "train loss" in line.lower():
                    raise Reject(
                        "malformed loss record at {}:{} ({})".format(
                            path, lineno, line.strip()[:80]
                        )
                    )
                continue
            loss_txt = m.group("loss").lower()
            if "nan" in loss_txt or "inf" in loss_txt:
                raise Reject(
                    "non-finite loss at {}:{} ({})".format(
                        path, lineno, m.group("loss")
                    )
                )
            total = int(m.group("total"))
            if total != expected_steps:
                raise Reject(
                    "step record says N/{} at {}:{} but expected N/{}".format(
                        total, path, lineno, expected_steps
                    )
                )
            step = int(m.group("step"))
            try:
                value = float(m.group("loss"))
            except ValueError:
                raise Reject(
                    "unparsable loss token at {}:{} ({})".format(
                        path, lineno, m.group("loss")
                    )
                )
            if not math.isfinite(value):
                raise Reject(
                    "non-finite loss value at {}:{} ({})".format(
                        path, lineno, m.group("loss")
                    )
                )
            if step in steps:
                raise Reject("duplicate step {} in {}".format(step, path))
            steps[step] = value
    if rank is None and len(ranks_seen) > 1:
        raise Reject(
            "ambiguous multi-rank log {} (ranks={}); pass --rank".format(
                path, sorted(ranks_seen)
            )
        )
    if rank is not None and ranks_seen and untagged_loss:
        raise Reject(
            "--rank {} given but {} loss record(s) carry no rank tag "
            "(first at {}:{}); refusing to mix tagged and untagged records".format(
                rank, len(untagged_loss), path, untagged_loss[0]
            )
        )
    if not steps:
        raise Reject("no parsable 'step N/M | train loss X' record in {}".format(path))

    want = list(range(step_base, step_base + expected_steps))
    missing = [s for s in want if s not in steps]
    extra = sorted(s for s in steps if s not in want)
    if missing:
        raise Reject("missing steps {} in {}".format(missing, path))
    if extra:
        raise Reject("unexpected steps {} in {}".format(extra, path))
    return {s: steps[s] for s in want}


def load_json(path, what):
    if not os.path.isfile(path):
        raise Reject("{} not found: {}".format(what, path))
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        raise Reject("{} is not valid JSON ({}): {}".format(what, exc, path))


def check_meta(meta, args, actual_log):
    required = [
        "case_id",
        "dtype",
        "exit_code",
        "steps",
        "log_sha256",
        "source_commit",
        "binary_sha256",
        "command",
        "hardware",
        "config",
    ]
    missing = [k for k in required if k not in meta]
    if missing:
        raise Reject("actual meta missing fields: {}".format(missing))
    if meta["dtype"] != args.dtype:
        raise Reject("dtype mismatch: meta={} cli={}".format(meta["dtype"], args.dtype))
    if args.case and meta["case_id"] != args.case:
        raise Reject(
            "case mismatch: meta={} expected={}".format(meta["case_id"], args.case)
        )
    ec = meta["exit_code"]
    if isinstance(ec, bool) or not isinstance(ec, int):
        raise Reject("training exit_code must be an integer, got {!r}".format(ec))
    if ec != 0:
        raise Reject("training exit_code={} (must be 0)".format(ec))
    steps_v = meta["steps"]
    if isinstance(steps_v, bool) or not isinstance(steps_v, int):
        raise Reject("meta steps must be an integer, got {!r}".format(steps_v))
    if steps_v != args.expected_steps:
        raise Reject("meta steps={} expected={}".format(steps_v, args.expected_steps))
    for key in ("source_commit", "binary_sha256", "command", "hardware"):
        val = meta[key]
        if (
            not isinstance(val, str)
            or not val.strip()
            or val.strip().lower() in ("unknown", "test", "todo")
        ):
            raise Reject(
                "meta field {} must be a real non-empty identifier (got {!r})".format(
                    key, val
                )
            )
    require_hex_token(meta["source_commit"], 40, "meta source_commit")
    require_hex_token(meta["binary_sha256"], 64, "meta binary_sha256")
    if not isinstance(meta["config"], dict):
        raise Reject(
            "actual meta config must be an object, got {}".format(
                type(meta["config"]).__name__
            )
        )
    real = sha256_file(actual_log)
    if meta["log_sha256"] != real:
        raise Reject(
            "actual log sha256 mismatch: meta={} file={}".format(
                meta["log_sha256"], real
            )
        )
    return real


NUMERIC_CONFIG_KEYS = (
    "batch_size",
    "sequence_length",
    "total_batch_size",
    "learning_rate",
)


def as_config_number(value, what):
    """Coerce a config field that is numeric by meaning, rejecting bool/str/None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Reject(
            "{} must be a number (bool/str/None not allowed), got {!r}".format(
                what, value
            )
        )
    number = float(value)
    if not math.isfinite(number):
        raise Reject("{} must be finite, got {!r}".format(what, value))
    return number


def require_hex_token(value, length, what):
    """Require a full-length lowercase/uppercase hex token, no trailing characters."""
    if not isinstance(value, str):
        raise Reject(
            "{} must be a {} chars hex string, got {!r}".format(what, length, value)
        )
    if not re.fullmatch(r"[0-9a-fA-F]{%d}" % length, value):
        raise Reject(
            "{} must be a {} chars hex string (got {!r})".format(what, length, value)
        )
    return value.lower()


def check_config(ref_cfg, actual_cfg, unknown_fields):
    """配置一致性：不一致即拒绝；无法确定的字段由 registry 显式列出 unknown。"""
    if not isinstance(ref_cfg, dict):
        raise Reject(
            "registry config must be an object, got {}".format(type(ref_cfg).__name__)
        )
    if not isinstance(actual_cfg, dict):
        raise Reject(
            "actual meta config must be an object, got {}".format(
                type(actual_cfg).__name__
            )
        )
    if not isinstance(unknown_fields, list) or any(
        not isinstance(item, str) for item in unknown_fields
    ):
        raise Reject("registry unknown_fields must be a list of strings")
    for key in ("weights_sha256", "data_sha256"):
        if key in unknown_fields:
            continue
        for label, cfg in (("registry", ref_cfg), ("actual", actual_cfg)):
            if key in cfg:
                require_hex_token(cfg[key], 64, "{} config {}".format(label, key))
    diffs = []
    for key in (
        "batch_size",
        "sequence_length",
        "total_batch_size",
        "learning_rate",
        "lr_schedule",
        "optimizer",
        "overfit_single_batch",
        "weights",
        "data",
        "weights_sha256",
        "data_sha256",
    ):
        if key in unknown_fields:
            continue
        if key in ref_cfg and key in actual_cfg:
            a, b = ref_cfg[key], actual_cfg[key]
            if key in NUMERIC_CONFIG_KEYS:
                # 按字段语义先验类型：bool/str/None 一律不得参与数值比较，
                # 不能靠 float() 转换后再看是否成功（那会抛出未捕获异常）。
                left = as_config_number(a, "registry config {}".format(key))
                right = as_config_number(b, "actual config {}".format(key))
                if abs(left - right) > 1e-12:
                    diffs.append("{}: ref={} actual={}".format(key, a, b))
                continue
            if a is None or b is None:
                raise Reject(
                    "{}: null is not a usable config value (ref={!r} actual={!r})".format(
                        key, a, b
                    )
                )
            if a != b:
                diffs.append("{}: ref={} actual={}".format(key, a, b))
        else:
            diffs.append(
                "{}: field absent on one side (treat as unknown -> not acceptance ready)".format(
                    key
                )
            )
    if diffs:
        raise Reject("config mismatch / unknown: " + "; ".join(diffs))


def compare(ref_steps, act_steps, dtype):
    rows = []
    worst = None
    for step in sorted(ref_steps):
        r, a = ref_steps[step], act_steps[step]
        abs_diff = abs(a - r)
        if dtype == "fp32":
            ok = abs_diff <= FP32_ATOL
            rel = abs_diff / abs(r) if r != 0 else (0.0 if a == 0 else math.inf)
            limit = FP32_ATOL
        else:
            if r == 0:
                rel = 0.0 if a == 0 else math.inf
            else:
                rel = abs_diff / abs(r)
            ok = rel <= BF16_RTOL
            limit = BF16_RTOL
            abs_diff = abs_diff
        rows.append(
            {
                "step": step,
                "ref": r,
                "actual": a,
                "abs_diff": abs_diff,
                "rel_diff": rel,
                "ok": ok,
                "limit": limit,
            }
        )
        metric = abs_diff if dtype == "fp32" else rel
        if worst is None or metric > worst[1]:
            worst = (step, metric, ok)
    failed = [row for row in rows if not row["ok"]]
    return rows, worst, failed


def main(argv=None):
    ap = argparse.ArgumentParser(description="strict loss acceptance (002-P1 v2 §A)")
    ap.add_argument("--actual", required=True)
    ap.add_argument("--actual-meta", required=True)
    ap.add_argument("--ref", help="reference log path (or use --registry with --case)")
    ap.add_argument("--registry")
    ap.add_argument("--case")
    ap.add_argument("--dtype", required=True, choices=["fp32", "bf16"])
    ap.add_argument("--expected-steps", type=int, default=10)
    ap.add_argument("--step-base", type=int, default=1)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--csv")
    ap.add_argument("--json")
    ap.add_argument(
        "--diagnostic",
        action="store_true",
        help="诊断模式：输入不完整时给出 DIAGNOSTIC_* 结论，绝不输出 RESULT: PASS",
    )
    ap.add_argument(
        "--allow-unregistered-ref",
        action="store_true",
        help="无 registry 时允许直接用 --ref 比较；结果标记 acceptance_ready=false",
    )
    args = ap.parse_args(argv)

    try:
        meta = load_json(args.actual_meta, "actual meta")
        if not isinstance(meta, dict):
            raise Reject(
                "actual meta top level must be an object, got {}".format(
                    type(meta).__name__
                )
            )
        actual_sha = check_meta(meta, args, args.actual)

        ref_cfg, unknown_fields, ref_info = {}, [], {}
        if args.registry:
            reg = load_json(args.registry, "registry")
            if not isinstance(reg, dict):
                raise Reject(
                    "registry top level must be an object, got {}".format(
                        type(reg).__name__
                    )
                )
            if not isinstance(reg.get("cases"), dict):
                raise Reject(
                    "registry cases must be an object, got {}".format(
                        type(reg.get("cases")).__name__
                    )
                )
            if not args.case:
                raise Reject("--registry requires --case")
            entry = reg["cases"].get(args.case)
            raw_unknown = (
                entry.get("unknown_fields", []) if isinstance(entry, dict) else []
            )
            if not isinstance(raw_unknown, list) or any(
                not isinstance(item, str) for item in raw_unknown
            ):
                raise Reject("registry unknown_fields must be a list of strings")
            if not isinstance(entry, dict):
                raise Reject("case {} not registered".format(args.case))
                raise Reject("case {} not registered".format(args.case))
            if entry.get("dtype") != args.dtype:
                raise Reject(
                    "registry dtype {} != cli {}".format(entry.get("dtype"), args.dtype)
                )
            if entry.get("exit_code") not in ("public_reference_exit_unknown", 0):
                raise Reject(
                    "registry exit_code must be 0 or public_reference_exit_unknown"
                )
            ref_hash = entry.get("log_sha256")
            if not isinstance(ref_hash, str) or len(ref_hash) != 64:
                raise Reject(
                    "formal mode needs a real reference log_sha256 (got {!r})".format(
                        ref_hash
                    )
                )
            ref_cfg = entry.get("config", {})
            unknown_fields = entry.get("unknown_fields", [])
            ref_info = {
                "kind": entry.get("reference_kind", "unknown"),
                "exit_code": entry.get("exit_code"),
                "log_sha256": entry.get("log_sha256", "unknown"),
            }
            reg_dir = os.path.dirname(os.path.abspath(args.registry))

            def reg_path(key):
                value = entry.get(key)
                if not isinstance(value, str) or not value:
                    return None
                if os.path.isabs(value):
                    return os.path.normpath(value)
                return os.path.normpath(os.path.join(reg_dir, value))

            ref_info = {
                "kind": entry.get("reference_kind", "unknown"),
                "role": entry.get("role", "primary"),
                "exit_code": entry.get("exit_code"),
                "log_sha256": ref_hash,
                "case_log_sha256": entry.get("case_log_sha256", "unknown"),
                "source_commit": entry.get("source_commit", "unknown"),
            }
            raw_file = reg_path("log_path")
            case_file = reg_path("case_log_path")

            if args.diagnostic:
                ref_file = args.ref or case_file or raw_file
                if ref_file is None:
                    raise Reject(
                        "diagnostic mode needs --ref or a registered reference path"
                    )
                if not os.path.isfile(ref_file):
                    raise Reject("reference file not found: {}".format(ref_file))
                ref_steps = parse_losses(
                    ref_file, args.expected_steps, args.step_base, args.rank
                )
            else:
                expected_file = case_file or raw_file
                if expected_file is None:
                    raise Reject(
                        "formal mode needs a registered reference file (log_path or case_log_path)"
                    )
                if args.ref and norm_path(args.ref) != norm_path(expected_file):
                    raise Reject(
                        "--ref conflicts with the registered reference path: --ref={} registry={}".format(
                            args.ref, expected_file
                        )
                    )
                ref_file = expected_file
                if not os.path.isfile(ref_file):
                    raise Reject("reference file not found: {}".format(ref_file))
                require_hex_token(
                    entry.get("source_commit"), 40, "registry source_commit"
                )
                if raw_file:
                    real_raw = sha256_file(raw_file)
                    if real_raw != ref_hash:
                        raise Reject(
                            "raw reference log sha256 mismatch: registry={} file={}".format(
                                ref_hash, real_raw
                            )
                        )
                if case_file:
                    case_hash = entry.get("case_log_sha256")
                    if not isinstance(case_hash, str) or not HEX64_RE.match(case_hash):
                        raise Reject(
                            "formal mode needs a real case_log_sha256 for the single-case reference (got {!r})".format(
                                case_hash
                            )
                        )
                    real_case = sha256_file(case_file)
                    if real_case != case_hash:
                        raise Reject(
                            "single-case reference sha256 mismatch: registry={} file={}".format(
                                case_hash, real_case
                            )
                        )
                    if not raw_file:
                        raise Reject(
                            "single-case reference without its originating raw log cannot be traced"
                        )
                    title = entry.get("case_extract_title")
                    if not isinstance(title, str) or not title.strip():
                        raise Reject(
                            "registry must record case_extract_title for the single-case derivation"
                        )
                    verify_derived_provenance(raw_file, case_file, title)
                ref_steps = parse_losses(
                    ref_file, args.expected_steps, args.step_base, args.rank
                )
                steps_reg = entry.get("steps")
                if steps_reg is not None:
                    if not isinstance(steps_reg, dict):
                        raise Reject("registry steps must be an object")
                    registered = {}
                    for key, value in steps_reg.items():
                        if not re.match(r"^[0-9]+$", str(key)):
                            raise Reject(
                                "registry step key {!r} is not a non-negative integer".format(
                                    key
                                )
                            )
                        if isinstance(value, bool) or not isinstance(
                            value, (int, float)
                        ):
                            raise Reject(
                                "registry step {}={!r} is not a number".format(
                                    key, value
                                )
                            )
                        if not math.isfinite(float(value)):
                            raise Reject("registry step {} is not finite".format(key))
                        registered[int(key)] = float(value)
                    want = list(
                        range(args.step_base, args.step_base + args.expected_steps)
                    )
                    missing = [x for x in want if x not in registered]
                    extra = sorted(x for x in registered if x not in want)
                    if missing or extra:
                        raise Reject(
                            "registry steps incomplete: missing={} extra={}".format(
                                missing, extra
                            )
                        )
                    for step in want:
                        if registered[step] != ref_steps[step]:
                            raise Reject(
                                "registry step {}={!r} differs from the frozen log value {!r}".format(
                                    step, registered[step], ref_steps[step]
                                )
                            )
        else:
            if not args.ref:
                raise Reject("need --ref or --registry/--case")
            if not args.allow_unregistered_ref:
                raise Reject(
                    "--ref without --registry is not acceptance-ready; "
                    "pass --allow-unregistered-ref for a diagnostic comparison only"
                )
            ref_steps = parse_losses(
                args.ref, args.expected_steps, args.step_base, args.rank
            )
            ref_info = {
                "kind": "unregistered_ref",
                "role": "diagnostic",
                "exit_code": "unknown",
            }

        if args.registry and unknown_fields:
            if not args.diagnostic:
                raise Reject(
                    "formal mode rejects unknown_fields={} (values usable for diagnostics only)".format(
                        unknown_fields
                    )
                )

        act_steps = parse_losses(
            args.actual, args.expected_steps, args.step_base, args.rank
        )
        if args.registry:
            check_config(ref_cfg, meta.get("config", {}), unknown_fields)
        else:
            ref_info.setdefault(
                "config_comparison", "skipped (unregistered ref, diagnostic only)"
            )
        rows, worst, failed = compare(ref_steps, act_steps, args.dtype)
    except Reject as exc:
        print("REJECT: {}".format(exc))
        print("RESULT: REJECTED (not a pass)")
        return 3

    payload = {
        "case_id": meta["case_id"],
        "dtype": args.dtype,
        "criterion": (
            "fp32 max_abs<=1e-5" if args.dtype == "fp32" else "bf16 max_rel<=1e-2"
        ),
        "criterion_source": "user_pdf_and_a100_log",
        "comparison_policy_source": "program_convention",
        "mode": "diagnostic" if args.diagnostic else "formal",
        "acceptance_ready": bool(args.registry)
        and not unknown_fields
        and not args.diagnostic,
        "reference": ref_info,
        "actual_log_sha256": actual_sha,
        "worst_step": worst[0] if worst else None,
        "worst_metric": worst[1] if worst else None,
        "steps": rows,
    }
    if args.json:
        # NOTE: a bf16 row whose reference is exactly 0 with a non-zero actual has
        # rel_diff = inf, so json.dump writes the non-standard token Infinity.
        # It never raises, but strict JSON parsers reject it; documented in
        # docs/validation.md instead of changing the output contract here.
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "ref",
                    "actual",
                    "abs_diff",
                    "rel_diff",
                    "ok",
                    "limit",
                ],
            )
            w.writeheader()
            w.writerows(rows)

    header = "{:>4} {:>14} {:>14} {:>12} {:>12} {:>6}".format(
        "step", "ref", "actual", "abs_diff", "rel_diff", "ok"
    )
    print(header)
    for row in rows:
        print(
            "{:>4} {:>14.6f} {:>14.6f} {:>12.3e} {:>12.3e} {:>6}".format(
                row["step"],
                row["ref"],
                row["actual"],
                row["abs_diff"],
                row["rel_diff"],
                "ok" if row["ok"] else "FAIL",
            )
        )
    print(
        "criterion: {} (source={})".format(
            payload["criterion"], payload["criterion_source"]
        )
    )
    print(
        "comparison policy: error denominator, zero-reference and <= boundary rules are program_convention"
    )
    print(
        "reference: {} role={} exit_code={} source_commit={}".format(
            ref_info.get("kind", "log"),
            ref_info.get("role", "primary"),
            ref_info.get("exit_code", "n/a"),
            ref_info.get("source_commit", "n/a"),
        )
    )
    print(
        "mode: {} acceptance_ready: {}".format(
            payload["mode"], payload["acceptance_ready"]
        )
    )

    if args.diagnostic or not payload["acceptance_ready"]:
        if failed:
            print(
                "DIAGNOSTIC_FAIL (worst step {} metric {:.3e}, {} of {} over limit); acceptance_ready=false".format(
                    worst[0], worst[1], len(failed), len(rows)
                )
            )
            return 2
        print(
            "DIAGNOSTIC_MATCH (worst step {} metric {:.3e}); acceptance_ready=false".format(
                worst[0], worst[1]
            )
        )
        return 0
    if failed:
        print(
            "RESULT: FAIL (worst step {} metric {:.3e}, {} of {} steps over limit)".format(
                worst[0], worst[1], len(failed), len(rows)
            )
        )
        return 2
    print("RESULT: PASS (worst step {} metric {:.3e})".format(worst[0], worst[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
