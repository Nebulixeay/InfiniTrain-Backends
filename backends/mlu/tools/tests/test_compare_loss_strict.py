#!/usr/bin/env python3
"""compare_loss_strict.py 的纯 host 自检（002-P1 v2 §A 清单 + 006-P1 v2 C3）。

只构造合成日志与合成登记，验证工具的通过/失败/拒绝/诊断行为；不产生任何训练结论。
用法：python backends/mlu/tools/tests/test_compare_loss_strict.py
退出码：0 全部符合预期；1 有不符合项。
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
TOOL = os.path.join(TOOLS, "compare_loss_strict.py")
PRODUCT_REGISTRY = os.path.join(TOOLS, "baseline_registry.json")
REFERENCES = os.path.join(TOOLS, "references")

STEPS = [
    5.356192,
    5.060585,
    4.860328,
    4.961310,
    4.888239,
    5.099205,
    4.860473,
    4.945070,
    5.121134,
    5.270753,
]
TITLE = "Running: gpt2_1 (fp32)"
SEP = "=" * 42 + "\n"
FORBIDDEN_REGISTRY_TOKENS = (
    "C:\\",
    "C:/",
    "Users",
    "kimi-code",
    "协作沟通",
    "资料/",
    "sessions",
)

CONFIG = {
    "batch_size": 4,
    "sequence_length": 64,
    "total_batch_size": 256,
    "learning_rate": 1e-4,
    "lr_schedule": "constant",
    "optimizer": "sgd",
    "overfit_single_batch": True,
    "weights": "gpt2_124M.bin",
    "data": "tiny_shakespeare_train.bin",
    "weights_sha256": "a" * 64,
    "data_sha256": "b" * 64,
}

results = []


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def write_log(path, steps, total=10, prefix=""):
    body = "".join(
        "{}step {:4d}/{} | train loss {} | lr 1.00e-04\n".format(prefix, i, total, v)
        for i, v in enumerate(steps, 1)
    )
    return write_text(path, body)


def case_segment(steps, title=TITLE, total=10):
    """The exact bytes of the frozen single-case file."""
    body = "".join(
        "step {:4d}/{} | train loss {}\n".format(i, total, v)
        for i, v in enumerate(steps, 1)
    )
    return title + "\n" + SEP + body


def write_pair(
    directory, steps=STEPS, total=10, title=TITLE, derived_override=None, raw_prefix=""
):
    """Write a raw multi-case log plus its single-case derivation; returns (raw, derived)."""
    derived = (
        case_segment(steps, title, total)
        if derived_override is None
        else derived_override
    )
    dpath = write_text(os.path.join(directory, "derived.log"), derived)
    extra_case = SEP + "Running: gpt2_1_bfloat16\n" + SEP + "bf16 filler\n"
    raw = (
        raw_prefix
        + SEP
        + derived
        + SEP
        + "Running: gpt2_1_bfloat16\n"
        + SEP
        + "bf16 filler\n"
    )
    rpath = write_text(os.path.join(directory, "raw.log"), raw)
    return rpath, dpath


def write_meta(path, log, **over):
    meta = {
        "case_id": "gpt2_1",
        "dtype": "fp32",
        "exit_code": 0,
        "steps": 10,
        "log_sha256": sha(log),
        "source_commit": "0" * 40,
        "binary_sha256": "f" * 64,
        "command": "synthetic-fixture cmd",
        "hardware": "synthetic-fixture",
        "config": dict(CONFIG),
    }
    meta.update(over)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return path


def write_registry(path, raw, derived, steps=STEPS, **over):
    if steps is None:
        steps_field = None
    elif isinstance(steps, dict):
        steps_field = dict(steps)
    else:
        steps_field = {str(i): v for i, v in enumerate(steps, 1)}
    entry = {
        "dtype": "fp32",
        "role": "primary",
        "reference_kind": "synthetic_fixture",
        "exit_code": "public_reference_exit_unknown",
        "source_commit": "0" * 40,
        "log_path": os.path.basename(raw),
        "log_sha256": sha(raw),
        "case_log_path": os.path.basename(derived),
        "case_log_sha256": sha(derived),
        "case_extract_title": TITLE,
        "steps": steps_field,
        "config": dict(CONFIG),
        "unknown_fields": [],
    }
    entry.update(over)
    payload = {"schema_version": 1, "cases": {"gpt2_1": entry}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def run(args, cwd=None):
    proc = subprocess.run(
        [sys.executable, TOOL] + args, capture_output=True, text=True, cwd=cwd
    )
    return proc.returncode, proc.stdout + proc.stderr


def case(name, expect_rc, args, expect_text=None, forbid_text=None, cwd=None):
    rc, out = run(args, cwd=cwd)
    ok = rc == expect_rc
    if expect_text is not None:
        ok = ok and expect_text in out
    if forbid_text is not None:
        ok = ok and forbid_text not in out
    ok = ok and "Traceback" not in out
    results.append(
        [name, ok, rc, expect_rc, out.strip().splitlines()[-1] if out.strip() else ""]
    )
    return out


def main():
    tmp = tempfile.mkdtemp(prefix="cls_selftest_")
    ref = write_log(os.path.join(tmp, "ref.log"), STEPS)
    raw, derived = write_pair(tmp)

    # ---- 原 002 轮 23 类 + C-005/C-006 回归（保留原用例名与预期） ----
    act = write_log(os.path.join(tmp, "act_ok.log"), STEPS)
    base_meta = write_meta(os.path.join(tmp, "meta_ok.json"), act)
    unreg = ["--ref", ref, "--allow-unregistered-ref", "--dtype", "fp32"]

    case(
        "normal pass",
        0,
        ["--actual", act, "--actual-meta", base_meta] + unreg,
        "DIAGNOSTIC_MATCH",
    )

    act_b = write_log(os.path.join(tmp, "act_bound.log"), [s + 1e-5 for s in STEPS])
    case(
        "boundary == 1e-5 passes",
        0,
        [
            "--actual",
            act_b,
            "--actual-meta",
            write_meta(os.path.join(tmp, "m1.json"), act_b),
        ]
        + unreg,
        None,
    )

    act_o = write_log(os.path.join(tmp, "act_over.log"), [s + 2e-5 for s in STEPS])
    case(
        "over threshold fails",
        2,
        [
            "--actual",
            act_o,
            "--actual-meta",
            write_meta(os.path.join(tmp, "m2.json"), act_o),
        ]
        + unreg,
        None,
    )
    case(
        "bf16 rel passes where fp32 fails",
        0,
        [
            "--actual",
            act_o,
            "--actual-meta",
            write_meta(os.path.join(tmp, "m3.json"), act_o, dtype="bf16"),
            "--ref",
            ref,
            "--allow-unregistered-ref",
            "--dtype",
            "bf16",
        ],
        None,
    )

    ref0 = write_log(os.path.join(tmp, "ref0.log"), [0.0] * 10)
    act0 = write_log(os.path.join(tmp, "act0.log"), [0.0] * 10)
    case(
        "zero ref and zero actual passes",
        0,
        [
            "--actual",
            act0,
            "--actual-meta",
            write_meta(os.path.join(tmp, "m4.json"), act0, dtype="bf16"),
            "--ref",
            ref0,
            "--allow-unregistered-ref",
            "--dtype",
            "bf16",
        ],
        None,
    )
    case(
        "zero ref but nonzero actual fails",
        2,
        [
            "--actual",
            act_o,
            "--actual-meta",
            write_meta(os.path.join(tmp, "m5.json"), act_o, dtype="bf16"),
            "--ref",
            ref0,
            "--allow-unregistered-ref",
            "--dtype",
            "bf16",
        ],
        None,
    )

    empty = write_text(os.path.join(tmp, "empty.log"), "")
    case(
        "empty log rejected",
        3,
        [
            "--actual",
            empty,
            "--actual-meta",
            write_meta(os.path.join(tmp, "m6.json"), empty),
        ]
        + unreg,
        "RESULT: REJECTED",
    )

    miss = write_log(os.path.join(tmp, "miss.log"), STEPS[:-1])
    case(
        "missing step rejected",
        3,
        [
            "--actual",
            miss,
            "--actual-meta",
            write_meta(os.path.join(tmp, "m7.json"), miss),
        ]
        + unreg,
        "missing steps",
    )
    dup = write_text(
        os.path.join(tmp, "dup.log"),
        "step 1/10 | train loss 5.0\nstep 1/10 | train loss 5.0\n",
    )
    case(
        "duplicate step rejected",
        3,
        [
            "--actual",
            dup,
            "--actual-meta",
            write_meta(os.path.join(tmp, "m8.json"), dup),
        ]
        + unreg,
        "duplicate step",
    )
    extra = write_log(os.path.join(tmp, "extra.log"), STEPS + [1.0], total=11)
    case(
        "extra step rejected",
        3,
        [
            "--actual",
            extra,
            "--actual-meta",
            write_meta(os.path.join(tmp, "m9.json"), extra),
        ]
        + unreg,
        None,
    )

    sci = os.path.join(tmp, "sci.log")
    write_text(
        sci,
        "".join(
            "step {:4d}/10 | train loss {:.6e}\n".format(i, v)
            for i, v in enumerate(STEPS, 1)
        ),
    )
    case(
        "scientific notation parsed",
        0,
        [
            "--actual",
            sci,
            "--actual-meta",
            write_meta(os.path.join(tmp, "m10.json"), sci),
        ]
        + unreg,
        "DIAGNOSTIC_MATCH",
    )

    for tag in ("nan", "inf"):
        p = write_text(
            os.path.join(tmp, "bad_{}.log".format(tag)),
            "".join(
                "step {:4d}/10 | train loss {}\n".format(i, tag if i == 3 else v)
                for i, v in enumerate(STEPS, 1)
            ),
        )
        case(
            "{} loss rejected".format(tag),
            3,
            [
                "--actual",
                p,
                "--actual-meta",
                write_meta(os.path.join(tmp, "m_{}.json".format(tag)), p),
            ]
            + unreg,
            "non-finite",
        )

    case(
        "meta exit_code!=0 rejected",
        3,
        [
            "--actual",
            act,
            "--actual-meta",
            write_meta(os.path.join(tmp, "m11.json"), act, exit_code=1),
        ]
        + unreg,
        "exit_code",
    )
    case(
        "meta dtype mismatch rejected",
        3,
        [
            "--actual",
            act,
            "--actual-meta",
            write_meta(os.path.join(tmp, "m12.json"), act, dtype="bf16"),
        ]
        + unreg,
        "dtype mismatch",
    )
    meta_no_field = write_meta(os.path.join(tmp, "m13.json"), act)
    with open(meta_no_field, encoding="utf-8") as f:
        meta_obj = json.load(f)
    meta_obj.pop("config")
    with open(meta_no_field, "w", encoding="utf-8") as f:
        json.dump(meta_obj, f)
    case(
        "meta missing field rejected",
        3,
        ["--actual", act, "--actual-meta", meta_no_field] + unreg,
        "missing fields",
    )
    case(
        "log sha mismatch rejected",
        3,
        [
            "--actual",
            act,
            "--actual-meta",
            write_meta(os.path.join(tmp, "m14.json"), act, log_sha256="deadbeef"),
        ]
        + unreg,
        "sha256 mismatch",
    )
    case(
        "case mismatch rejected",
        3,
        [
            "--actual",
            act,
            "--actual-meta",
            base_meta,
            "--ref",
            ref,
            "--allow-unregistered-ref",
            "--dtype",
            "fp32",
            "--case",
            "gpt2_2",
        ],
        "case mismatch",
    )

    multi = os.path.join(tmp, "multi.log")
    write_text(
        multi,
        "".join(
            "Rank 0: step {:4d}/10 | train loss {}\nRank 1: step {:4d}/10 | train loss {}\n".format(
                i, v, i, v
            )
            for i, v in enumerate(STEPS, 1)
        ),
    )
    case(
        "multi-rank ambiguous rejected",
        3,
        [
            "--actual",
            multi,
            "--actual-meta",
            write_meta(os.path.join(tmp, "m16.json"), multi),
        ]
        + unreg,
        "ambiguous multi-rank",
    )
    case(
        "multi-rank with --rank passes",
        0,
        [
            "--actual",
            multi,
            "--actual-meta",
            write_meta(os.path.join(tmp, "m17.json"), multi),
            "--ref",
            ref,
            "--allow-unregistered-ref",
            "--dtype",
            "fp32",
            "--rank",
            "0",
        ],
        None,
    )

    reg_cfg = write_registry(os.path.join(tmp, "registry_cfg.json"), raw, derived)
    case(
        "config mismatch rejected (registry)",
        3,
        [
            "--actual",
            act,
            "--actual-meta",
            write_meta(os.path.join(tmp, "m18.json"), act, config={"batch_size": 8}),
            "--registry",
            reg_cfg,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        None,
    )

    reg = write_registry(os.path.join(tmp, "registry.json"), raw, derived)
    formal = [
        "--actual",
        act,
        "--actual-meta",
        base_meta,
        "--registry",
        reg,
        "--case",
        "gpt2_1",
        "--dtype",
        "fp32",
    ]
    case("registry unknown exit_code accepted", 0, list(formal), "RESULT: PASS")
    case(
        "registry unknown field -> not acceptance ready",
        3,
        [
            "--actual",
            act,
            "--actual-meta",
            write_meta(os.path.join(tmp, "m20.json"), act, config={"batch_size": 4}),
            "--registry",
            reg,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        None,
    )

    case(
        "A-R1 registry missing step rejected",
        3,
        [
            "--registry",
            write_registry(
                os.path.join(tmp, "r_miss.json"),
                raw,
                derived,
                steps={str(i): v for i, v in list(enumerate(STEPS, 1))[:-1]},
            ),
            "--actual",
            act,
            "--actual-meta",
            base_meta,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "registry steps incomplete",
    )
    case(
        "A-R2 unknown_fields rejected in formal mode",
        3,
        [
            "--registry",
            write_registry(
                os.path.join(tmp, "r_unk.json"),
                raw,
                derived,
                unknown_fields=["weights", "data_hash"],
            ),
            "--actual",
            act,
            "--actual-meta",
            base_meta,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "rejects unknown_fields",
    )
    case(
        "A-R2 unknown reference hash rejected",
        3,
        [
            "--registry",
            write_registry(
                os.path.join(tmp, "r_hash.json"), raw, derived, log_sha256="unknown"
            ),
            "--actual",
            act,
            "--actual-meta",
            base_meta,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "real reference log_sha256",
    )
    meta_no_prov = write_meta(os.path.join(tmp, "m22.json"), act)
    with open(meta_no_prov, encoding="utf-8") as f:
        meta_obj = json.load(f)
    meta_obj.pop("command")
    with open(meta_no_prov, "w", encoding="utf-8") as f:
        json.dump(meta_obj, f)
    case(
        "A-R2 missing provenance rejected",
        3,
        [
            "--registry",
            reg,
            "--actual",
            act,
            "--actual-meta",
            meta_no_prov,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "missing fields",
    )
    case(
        "A-R2 weights mismatch rejected",
        3,
        [
            "--registry",
            reg,
            "--actual",
            act,
            "--actual-meta",
            write_meta(
                os.path.join(tmp, "m23.json"),
                act,
                config=dict(CONFIG, weights="other.bin"),
            ),
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "weights",
    )
    garbage = write_text(
        os.path.join(tmp, "garbage.log"),
        "".join(
            "step {:4d}/10 | train loss {}\n".format(i, "5.0e+garbage" if i == 1 else v)
            for i, v in enumerate(STEPS, 1)
        ),
    )
    case(
        "A-R3 garbage exponent rejected",
        3,
        [
            "--registry",
            reg,
            "--actual",
            garbage,
            "--actual-meta",
            write_meta(os.path.join(tmp, "m24.json"), garbage),
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        None,
    )
    case("A-R2 formal pass still works", 0, list(formal), "RESULT: PASS")
    case(
        "A-R2 diagnostic mode never prints PASS",
        0,
        [
            "--registry",
            write_registry(
                os.path.join(tmp, "r_unk2.json"),
                raw,
                derived,
                unknown_fields=["weights"],
            ),
            "--actual",
            act,
            "--actual-meta",
            base_meta,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
            "--diagnostic",
        ],
        "DIAGNOSTIC_MATCH",
        forbid_text="RESULT: PASS",
    )

    # ---- 006-P1 v2 C3：新增反例与可移植性 ----
    disagree = {str(i): v for i, v in enumerate(STEPS, 1)}
    disagree["10"] = 5.999999
    case(
        "C3 registry disagrees with file (no --ref)",
        3,
        [
            "--registry",
            write_registry(
                os.path.join(tmp, "r_dis.json"), raw, derived, steps=disagree
            ),
            "--actual",
            act,
            "--actual-meta",
            base_meta,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "differs from the frozen log",
    )
    case(
        "C3 registry disagrees with file (with --ref)",
        3,
        [
            "--registry",
            write_registry(
                os.path.join(tmp, "r_dis2.json"), raw, derived, steps=disagree
            ),
            "--ref",
            derived,
            "--actual",
            act,
            "--actual-meta",
            base_meta,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "differs from the frozen log",
    )
    case(
        "C3 registry extra step rejected",
        3,
        [
            "--registry",
            write_registry(
                os.path.join(tmp, "r_extra.json"),
                raw,
                derived,
                steps=dict({str(i): v for i, v in enumerate(STEPS, 1)}, **{"11": 1.0}),
            ),
            "--actual",
            act,
            "--actual-meta",
            base_meta,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "registry steps incomplete",
    )
    case(
        "C3 extra decimal in registry rejected",
        3,
        [
            "--registry",
            write_registry(
                os.path.join(tmp, "r_dec.json"),
                raw,
                derived,
                steps=dict(
                    {str(i): v for i, v in enumerate(STEPS, 1)}, **{"1": 5.3561921}
                ),
            ),
            "--actual",
            act,
            "--actual-meta",
            base_meta,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "differs from the frozen log",
    )
    dup_dir = tempfile.mkdtemp(prefix="cls_dup_", dir=tmp)
    raw_dup, derived_dup = write_pair(dup_dir, raw_prefix=SEP + case_segment(STEPS))
    case(
        "C3 duplicate case title rejected",
        3,
        [
            "--registry",
            write_registry(os.path.join(dup_dir, "r.json"), raw_dup, derived_dup),
            "--actual",
            act,
            "--actual-meta",
            write_meta(os.path.join(tmp, "m25.json"), act),
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "appears 2 times",
    )
    tamper_dir = tempfile.mkdtemp(prefix="cls_tamper_", dir=tmp)
    raw_t, derived_t = write_pair(tamper_dir)
    write_text(derived_t, case_segment([s + 1e-6 for s in STEPS]))
    case(
        "C3 derived content differs from raw segment",
        3,
        [
            "--registry",
            write_registry(os.path.join(tamper_dir, "r.json"), raw_t, derived_t),
            "--actual",
            act,
            "--actual-meta",
            write_meta(os.path.join(tmp, "m26.json"), act),
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "does not match the case segment",
    )
    case(
        "C3 raw hash mismatch rejected",
        3,
        [
            "--registry",
            write_registry(
                os.path.join(tmp, "r_rawhash.json"), raw, derived, log_sha256="0" * 64
            ),
            "--actual",
            act,
            "--actual-meta",
            base_meta,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "raw reference log sha256 mismatch",
    )
    case(
        "C3 derived hash mismatch rejected",
        3,
        [
            "--registry",
            write_registry(
                os.path.join(tmp, "r_dhash.json"),
                raw,
                derived,
                case_log_sha256="0" * 64,
            ),
            "--actual",
            act,
            "--actual-meta",
            base_meta,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "single-case reference sha256 mismatch",
    )
    case(
        "C3 derived without raw cannot be traced",
        3,
        [
            "--registry",
            write_registry(
                os.path.join(tmp, "r_noraw.json"), raw, derived, log_path=None
            ),
            "--actual",
            act,
            "--actual-meta",
            base_meta,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "cannot be traced",
    )
    case(
        "C3 --ref conflicts with registry path",
        3,
        [
            "--registry",
            reg,
            "--ref",
            ref,
            "--actual",
            act,
            "--actual-meta",
            base_meta,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "--ref conflicts",
    )
    case(
        "C3 --ref same file, different spelling accepted",
        0,
        [
            "--registry",
            reg,
            "--ref",
            os.path.join(tmp, ".", "derived.log"),
            "--actual",
            act,
            "--actual-meta",
            base_meta,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "RESULT: PASS",
    )
    case(
        "C3 registry without steps still validates file",
        0,
        [
            "--registry",
            write_registry(
                os.path.join(tmp, "r_nosteps.json"), raw, derived, steps=None
            ),
            "--actual",
            act,
            "--actual-meta",
            base_meta,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "RESULT: PASS",
    )

    # 诊断模式：JSON 标记与终端一致
    diag_json = os.path.join(tmp, "diag.json")
    out = case(
        "C3 diagnostic JSON acceptance_ready false",
        0,
        [
            "--registry",
            write_registry(
                os.path.join(tmp, "r_diag.json"),
                raw,
                derived,
                unknown_fields=["weights"],
            ),
            "--actual",
            act,
            "--actual-meta",
            base_meta,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
            "--diagnostic",
            "--json",
            diag_json,
        ],
        "DIAGNOSTIC_MATCH",
        forbid_text="RESULT: PASS",
    )
    with open(diag_json, encoding="utf-8") as f:
        diag_payload = json.load(f)
    ok = (
        diag_payload["acceptance_ready"] is False
        and diag_payload["mode"] == "diagnostic"
        and diag_payload["criterion_source"] == "user_pdf_and_a100_log"
    )
    results.append(
        [
            "C3 diagnostic JSON flags consistent",
            ok,
            0,
            0,
            json.dumps(
                {
                    k: diag_payload[k]
                    for k in ("acceptance_ready", "mode", "criterion_source")
                }
            ),
        ]
    )

    # 正式模式 JSON：判据来源拆开标注
    formal_json = os.path.join(tmp, "formal.json")
    case(
        "C5 formal JSON sources split",
        0,
        list(formal) + ["--json", formal_json],
        "RESULT: PASS",
    )
    with open(formal_json, encoding="utf-8") as f:
        formal_payload = json.load(f)
    ok = (
        formal_payload["criterion_source"] == "user_pdf_and_a100_log"
        and formal_payload["comparison_policy_source"] == "program_convention"
        and formal_payload["acceptance_ready"] is True
        and formal_payload["reference"]["source_commit"].startswith("5a0f27a5") is False
    )
    results.append(
        [
            "C5 formal JSON sources split",
            ok,
            0,
            0,
            json.dumps(
                {
                    k: formal_payload[k]
                    for k in (
                        "criterion_source",
                        "comparison_policy_source",
                        "acceptance_ready",
                    )
                }
            ),
        ]
    )

    # JSON 结构/类型错误：不产生 Traceback
    bad_meta = write_meta(os.path.join(tmp, "m_bad.json"), act)
    with open(bad_meta, encoding="utf-8") as f:
        bad_obj = json.load(f)
    bad_obj["steps"] = None
    with open(bad_meta, "w", encoding="utf-8") as f:
        json.dump(bad_obj, f)
    case(
        "C3 meta steps null rejected without traceback",
        3,
        ["--actual", act, "--actual-meta", bad_meta] + unreg,
        "integer",
    )
    bad_reg = write_text(os.path.join(tmp, "bad_registry.json"), "{not json")
    case(
        "C3 registry invalid JSON rejected",
        3,
        [
            "--registry",
            bad_reg,
            "--actual",
            act,
            "--actual-meta",
            base_meta,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "not valid JSON",
    )

    # E3：bf16 零参考导致无穷，JSON 可写出且无 Traceback（非标准 Infinity 已记录）
    inf_json = os.path.join(tmp, "inf.json")
    case(
        "E3 bf16 zero-ref infinity writes JSON without traceback",
        2,
        [
            "--actual",
            act_o,
            "--actual-meta",
            write_meta(os.path.join(tmp, "m_inf.json"), act_o, dtype="bf16"),
            "--ref",
            ref0,
            "--allow-unregistered-ref",
            "--dtype",
            "bf16",
            "--json",
            inf_json,
        ],
        "DIAGNOSTIC_FAIL",
    )
    with open(inf_json, encoding="utf-8") as f:
        inf_text = f.read()
    results.append(
        [
            "E3 bf16 infinity token documented",
            "Infinity" in inf_text,
            0,
            0,
            "Infinity present" if "Infinity" in inf_text else "no Infinity token",
        ]
    )

    # C1：产品 registry 不含本机/跨仓库路径
    with open(PRODUCT_REGISTRY, encoding="utf-8") as f:
        reg_text = f.read()
    leaked = [token for token in FORBIDDEN_REGISTRY_TOKENS if token in reg_text]
    reg_obj = json.loads(reg_text)
    entry = reg_obj["cases"]["gpt2_1"]
    abs_paths = [
        value
        for key, value in entry.items()
        if key.endswith("_path") and isinstance(value, str) and os.path.isabs(value)
    ]
    ok = not leaked and not abs_paths
    results.append(
        [
            "C1 product registry has no private paths",
            ok,
            0,
            0,
            "leaked={} abs={}".format(leaked, abs_paths),
        ]
    )

    # C1 + 真实 registry：正式模式必须先通过 raw/derived/来源校验，再因 unknown 资产拒绝
    case(
        "C1 product registry formal run stops at unknown assets",
        3,
        [
            "--registry",
            PRODUCT_REGISTRY,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
            "--actual",
            act,
            "--actual-meta",
            base_meta,
        ],
        "rejects unknown_fields",
    )

    # C1 可移植性：整目录复制到别处，从不同 cwd 运行
    copy_root = tempfile.mkdtemp(prefix="cls_copy_", dir=tmp)
    copy_tools = os.path.join(copy_root, "tools")
    shutil.copytree(TOOLS, copy_tools)
    elsewhere = tempfile.mkdtemp(prefix="cls_cwd_", dir=tmp)
    sys_raw, sys_derived = write_pair(
        os.path.join(copy_tools, "references"), title=TITLE
    )
    copy_reg = write_registry(
        os.path.join(copy_tools, "portable_registry.json"),
        os.path.join(copy_tools, "references", "raw.log"),
        os.path.join(copy_tools, "references", "derived.log"),
    )
    with open(copy_reg, encoding="utf-8") as f:
        copy_obj = json.load(f)
    copy_entry = copy_obj["cases"]["gpt2_1"]
    copy_entry["log_path"] = "references/raw.log"
    copy_entry["case_log_path"] = "references/derived.log"
    with open(copy_reg, "w", encoding="utf-8") as f:
        json.dump(copy_obj, f)
    case(
        "C1 relocated tools dir runs from another cwd",
        0,
        [
            "--registry",
            copy_reg,
            "--actual",
            act,
            "--actual-meta",
            base_meta,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "RESULT: PASS",
        cwd=elsewhere,
    )

    # ---- C-004：Codex 独立 CLI 审计发现的 8 条真实入口反例 ----
    case(
        "C4 invalid source_commit format rejected",
        3,
        [
            "--actual",
            act,
            "--actual-meta",
            write_meta(
                os.path.join(tmp, "m_sc.json"), act, source_commit="not-a-commit"
            ),
        ]
        + unreg,
        "source_commit",
    )
    case(
        "C4 invalid binary_sha256 format rejected",
        3,
        [
            "--actual",
            act,
            "--actual-meta",
            write_meta(
                os.path.join(tmp, "m_bh.json"), act, binary_sha256="invalid-hash"
            ),
        ]
        + unreg,
        "binary_sha256",
    )
    case(
        "C4 invalid but matching asset hashes rejected",
        3,
        [
            "--registry",
            write_registry(
                os.path.join(tmp, "r_badasset.json"),
                raw,
                derived,
                config=dict(
                    CONFIG, weights_sha256="not-a-hash", data_sha256="not-a-hash"
                ),
            ),
            "--actual",
            act,
            "--actual-meta",
            write_meta(
                os.path.join(tmp, "m_badasset.json"),
                act,
                config=dict(
                    CONFIG, weights_sha256="not-a-hash", data_sha256="not-a-hash"
                ),
            ),
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "weights_sha256",
    )
    case(
        "C4 null config rejected without traceback",
        3,
        [
            "--actual",
            act,
            "--actual-meta",
            write_meta(os.path.join(tmp, "m_nullcfg.json"), act, config=None),
        ]
        + unreg,
        "config must be an object",
    )
    case(
        "C4 registry top-level list rejected",
        3,
        [
            "--registry",
            write_text(os.path.join(tmp, "r_toplevel.json"), "[]"),
            "--actual",
            act,
            "--actual-meta",
            base_meta,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "top level must be an object",
    )
    case(
        "C4 registry cases null rejected",
        3,
        [
            "--registry",
            write_text(
                os.path.join(tmp, "r_casesnull.json"),
                json.dumps({"schema_version": 1, "cases": None}),
            ),
            "--actual",
            act,
            "--actual-meta",
            base_meta,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "cases must be an object",
    )
    with open(reg, encoding="utf-8") as f:
        reg_obj = json.load(f)
    reg_obj["cases"]["gpt2_1"]["unknown_fields"] = None
    case(
        "C4 registry unknown_fields null rejected",
        3,
        [
            "--registry",
            write_text(os.path.join(tmp, "r_unknull.json"), json.dumps(reg_obj)),
            "--actual",
            act,
            "--actual-meta",
            base_meta,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "unknown_fields must be a list of strings",
    )
    mixed = os.path.join(tmp, "mixed_rank.log")
    write_text(
        mixed,
        "".join(
            (
                "Rank 0: step {:4d}/10 | train loss {}\n".format(i, v)
                if i <= 5
                else "step {:4d}/10 | train loss {}\n".format(i, v)
            )
            for i, v in enumerate(STEPS, 1)
        ),
    )
    case(
        "C4 explicit rank mixing untagged loss rejected",
        3,
        [
            "--actual",
            mixed,
            "--actual-meta",
            write_meta(os.path.join(tmp, "m_mixed.json"), mixed),
            "--ref",
            ref,
            "--allow-unregistered-ref",
            "--dtype",
            "fp32",
            "--rank",
            "0",
        ],
        "no rank tag",
    )

    # ---- C-005 补充复核的 2 个拒绝漏洞 + 同源加强 ----
    case(
        "C5 learning_rate wrong type rejected",
        3,
        [
            "--registry",
            reg,
            "--actual",
            act,
            "--actual-meta",
            write_meta(
                os.path.join(tmp, "m_lr_bad.json"),
                act,
                config=dict(CONFIG, learning_rate="not-a-number"),
            ),
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "learning_rate must be a number",
    )
    case(
        "C5 learning_rate same bad string on both sides rejected",
        3,
        [
            "--registry",
            write_registry(
                os.path.join(tmp, "r_lr_bad.json"),
                raw,
                derived,
                config=dict(CONFIG, learning_rate="not-a-number"),
            ),
            "--actual",
            act,
            "--actual-meta",
            write_meta(
                os.path.join(tmp, "m_lr_bad2.json"),
                act,
                config=dict(CONFIG, learning_rate="not-a-number"),
            ),
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "learning_rate must be a number",
    )
    case(
        "C5 reference source_commit bad format rejected",
        3,
        [
            "--registry",
            write_registry(
                os.path.join(tmp, "r_commit_bad.json"),
                raw,
                derived,
                source_commit="not-a-commit",
            ),
            "--actual",
            act,
            "--actual-meta",
            base_meta,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "registry source_commit",
    )
    case(
        "C5 reference source_commit missing rejected",
        3,
        [
            "--registry",
            write_registry(
                os.path.join(tmp, "r_commit_none.json"),
                raw,
                derived,
                source_commit=None,
            ),
            "--actual",
            act,
            "--actual-meta",
            base_meta,
            "--case",
            "gpt2_1",
            "--dtype",
            "fp32",
        ],
        "registry source_commit",
    )

    bad = [r for r in results if not r[1]]
    for name, ok, rc, expect, last in results:
        print(
            "{:<48} {:<5} rc={} expect={}  {}".format(
                name, "ok" if ok else "FAIL", rc, expect, last
            )
        )
    print(
        "compare_loss_strict selftest: {} ({}/{} cases)".format(
            "PASS" if not bad else "FAIL", len(results) - len(bad), len(results)
        )
    )
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
