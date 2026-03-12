#!/usr/bin/env python3
"""Three-stage fast-to-formal search for C > B settings.

Stage-1: B-only quick sweep.
Stage-2: Full A/B/C on top-k configs, keep configs where C improves B on rot/trans/CD.
Stage-3: Formal recheck on best 1-2 configs with larger sample and submap_radius=1.
"""

from __future__ import annotations

import itertools
import json
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tests" / "tests_result" / "hybrid_registration_7scenes" / "three_stage_search"
DATASET_ROOT = Path("/home/hba/Documents/Dataset/7_scenes")
SCENES = ["office", "redkitchen"]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_registration_module():
    module_path = ROOT / "tests" / "test_hybrid_registration_robustness_7scenes.py"
    spec = importlib.util.spec_from_file_location("hybrid_reg_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reg = load_registration_module()


def load_model() -> Any:
    ckpt_path = ROOT / "ckpt" / "model_tracker_fixed_e20.pt"
    model = reg.VGGT(merging=None, merge_ratio=0.9, vis_attn_map=False)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.cuda().eval()
    for param in model.parameters():
        if param.dtype != torch.float32:
            param.data = param.data.float()
    return model


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("method", as_index=False)
        .agg(
            mean_rot=("rotation_error_deg", "mean"),
            mean_trans=("translation_error_m", "mean"),
            mean_cd=("chamfer_distance", "mean"),
            mean_rt=("runtime_ms", "mean"),
            recall=("recall_hit", "mean"),
            pairs=("method", "size"),
        )
        .sort_values("method")
    )


def eval_config(model: Any, cfg: Dict[str, float], methods: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for scene in SCENES:
        pairs = reg.collect_pairs(
            model=model,
            dataset_root=DATASET_ROOT,
            scene=scene,
            frame_stride=int(cfg["frame_stride"]),
            min_rot_deg=float(cfg["min_rot_deg"]),
            min_trans_m=float(cfg["min_trans_m"]),
            max_rot_deg=float(cfg["max_rot_deg"]),
            max_trans_m=float(cfg["max_trans_m"]),
            max_gt_cd=float(cfg["max_gt_cd"]),
            max_pairs_per_seq=int(cfg["max_pairs_per_seq"]),
            submap_radius=int(cfg["submap_radius"]),
            min_desc_sim=float(cfg["min_desc_sim"]),
        )
        for pair in pairs:
            rows.extend(reg.evaluate_pair(pair, methods=methods))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def stage1(model: Any) -> pd.DataFrame:
    grid = {
        "min_rot_deg": [10.0, 14.0],
        "min_trans_m": [0.20, 0.30],
        "max_rot_deg": [38.0, 48.0],
        "max_trans_m": [0.75, 0.95],
        "max_gt_cd": [0.20, 0.30],
        "min_desc_sim": [0.50],
    }

    fixed = {
        "frame_stride": 25,
        "max_pairs_per_seq": 4,
        "submap_radius": 0,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    for idx, vals in enumerate(combos, start=1):
        cfg = dict(zip(keys, vals))
        cfg.update(fixed)
        print(f"[Stage1] config {idx}/{len(combos)}: {cfg}", flush=True)
        df = eval_config(model, cfg, methods=["B"])
        if df.empty:
            continue
        b = summarize(df).iloc[0]
        score = 0.45 * b["mean_cd"] + 0.30 * b["mean_rot"] + 0.25 * b["mean_trans"]
        records.append(
            {
                **cfg,
                "pairs": int(b["pairs"]),
                "B_rot": float(b["mean_rot"]),
                "B_trans": float(b["mean_trans"]),
                "B_cd": float(b["mean_cd"]),
                "B_rt": float(b["mean_rt"]),
                "B_recall": float(b["recall"]),
                "stage1_score": float(score),
            }
        )
        print(f"[Stage1] done {idx}/{len(combos)}, valid={len(records)}", flush=True)

    out = pd.DataFrame(records).sort_values("stage1_score", ascending=True)
    out.to_csv(OUT_DIR / "stage1_b_only_ranking.csv", index=False)
    return out


def stage2(model: Any, stage1_rank: pd.DataFrame, topk: int = 6) -> pd.DataFrame:
    chosen = stage1_rank.head(topk).to_dict("records")
    records = []

    for i, cfg in enumerate(chosen, start=1):
        cfg_eval = {
            "frame_stride": int(cfg["frame_stride"]),
            "min_rot_deg": float(cfg["min_rot_deg"]),
            "min_trans_m": float(cfg["min_trans_m"]),
            "max_rot_deg": float(cfg["max_rot_deg"]),
            "max_trans_m": float(cfg["max_trans_m"]),
            "max_gt_cd": float(cfg["max_gt_cd"]),
            "max_pairs_per_seq": int(cfg["max_pairs_per_seq"]),
            "submap_radius": int(cfg["submap_radius"]),
            "min_desc_sim": float(cfg["min_desc_sim"]),
        }
        df = eval_config(model, cfg_eval, methods=["A", "B", "C"])
        if df.empty:
            continue
        s = summarize(df).set_index("method")
        if not {"B", "C"}.issubset(set(s.index)):
            continue

        b = s.loc["B"]
        c = s.loc["C"]
        c_better_all = bool((c["mean_rot"] < b["mean_rot"]) and (c["mean_trans"] < b["mean_trans"]) and (c["mean_cd"] < b["mean_cd"]))

        records.append(
            {
                **cfg_eval,
                "pairs": int(s["pairs"].sum() // 3),
                "B_rot": float(b["mean_rot"]),
                "C_rot": float(c["mean_rot"]),
                "B_trans": float(b["mean_trans"]),
                "C_trans": float(c["mean_trans"]),
                "B_cd": float(b["mean_cd"]),
                "C_cd": float(c["mean_cd"]),
                "B_rt": float(b["mean_rt"]),
                "C_rt": float(c["mean_rt"]),
                "rot_gain_pct": float((b["mean_rot"] - c["mean_rot"]) / (b["mean_rot"] + 1e-8) * 100.0),
                "trans_gain_pct": float((b["mean_trans"] - c["mean_trans"]) / (b["mean_trans"] + 1e-8) * 100.0),
                "cd_gain_pct": float((b["mean_cd"] - c["mean_cd"]) / (b["mean_cd"] + 1e-8) * 100.0),
                "c_better_all": c_better_all,
            }
        )
        print(f"[Stage2] {i}/{len(chosen)} done, c_better_all={c_better_all}")

    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values(["c_better_all", "cd_gain_pct", "rot_gain_pct", "trans_gain_pct"], ascending=[False, False, False, False])
    out.to_csv(OUT_DIR / "stage2_top_configs_full_abc.csv", index=False)
    out[out["c_better_all"]].to_csv(OUT_DIR / "stage2_candidates_c_better_all.csv", index=False)
    return out


def stage3(model: Any, stage2_rank: pd.DataFrame, max_candidates: int = 1) -> pd.DataFrame:
    candidates = stage2_rank[stage2_rank["c_better_all"]].head(max_candidates)
    if candidates.empty:
        candidates = stage2_rank.head(max_candidates)

    records = []
    for i, row in enumerate(candidates.to_dict("records"), start=1):
        cfg_eval = {
            "frame_stride": int(row["frame_stride"]),
            "min_rot_deg": float(row["min_rot_deg"]),
            "min_trans_m": float(row["min_trans_m"]),
            "max_rot_deg": float(row["max_rot_deg"]),
            "max_trans_m": float(row["max_trans_m"]),
            "max_gt_cd": float(row["max_gt_cd"]),
            "max_pairs_per_seq": 12,
            "submap_radius": 1,
            "min_desc_sim": float(row["min_desc_sim"]),
        }
        df = eval_config(model, cfg_eval, methods=["A", "B", "C"])
        if df.empty:
            continue
        s = summarize(df).set_index("method")
        b = s.loc["B"]
        c = s.loc["C"]
        records.append(
            {
                **cfg_eval,
                "pairs": int(s["pairs"].sum() // 3),
                "B_rot": float(b["mean_rot"]),
                "C_rot": float(c["mean_rot"]),
                "B_trans": float(b["mean_trans"]),
                "C_trans": float(c["mean_trans"]),
                "B_cd": float(b["mean_cd"]),
                "C_cd": float(c["mean_cd"]),
                "B_rt": float(b["mean_rt"]),
                "C_rt": float(c["mean_rt"]),
                "rot_gain_pct": float((b["mean_rot"] - c["mean_rot"]) / (b["mean_rot"] + 1e-8) * 100.0),
                "trans_gain_pct": float((b["mean_trans"] - c["mean_trans"]) / (b["mean_trans"] + 1e-8) * 100.0),
                "cd_gain_pct": float((b["mean_cd"] - c["mean_cd"]) / (b["mean_cd"] + 1e-8) * 100.0),
                "c_better_all": bool((c["mean_rot"] < b["mean_rot"]) and (c["mean_trans"] < b["mean_trans"]) and (c["mean_cd"] < b["mean_cd"])),
            }
        )
        print(f"[Stage3] {i}/{len(candidates)} done")

    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values(["c_better_all", "cd_gain_pct", "rot_gain_pct", "trans_gain_pct"], ascending=[False, False, False, False])
    out.to_csv(OUT_DIR / "stage3_formal_recheck.csv", index=False)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model()

    s1 = stage1(model)
    if s1.empty:
        print("No valid stage-1 configs.")
        return

    s2 = stage2(model, s1, topk=6)
    if s2.empty:
        print("No valid stage-2 configs.")
        return

    s3 = stage3(model, s2, max_candidates=1)

    report = {
        "stage1_count": int(len(s1)),
        "stage2_count": int(len(s2)),
        "stage2_c_better_all": int(s2["c_better_all"].sum()) if not s2.empty else 0,
        "stage3_count": int(len(s3)) if not s3.empty else 0,
        "stage3_c_better_all": int(s3["c_better_all"].sum()) if not s3.empty else 0,
    }
    with (OUT_DIR / "three_stage_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Saved:")
    print(OUT_DIR / "stage1_b_only_ranking.csv")
    print(OUT_DIR / "stage2_top_configs_full_abc.csv")
    print(OUT_DIR / "stage2_candidates_c_better_all.csv")
    print(OUT_DIR / "stage3_formal_recheck.csv")
    print(OUT_DIR / "three_stage_report.json")


if __name__ == "__main__":
    main()
