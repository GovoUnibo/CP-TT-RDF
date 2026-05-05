#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from benchmark.panda_benchmark_core import BenchConfig, run_benchmark_suite

RUN_TRAIN_WEIGHTS = False
INCLUDE_WEIGHTS_METHOD = False
CP_TT_SOURCE = "train"  # train diretto da dataset, senza passare da weights

TRAIN_ITERS_WEIGHTS = 0
TRAIN_ITERS_CP = 1200
TRAIN_ITERS_TT = 3000

CP_METHOD = "adam"
TT_METHOD = "adam"
CP_LR = 5e-4
TT_LR = 2e-4
CP_RIDGE = 2e-4
TT_RIDGE = 2e-4
CP_BATCH_SIZE = 32_768
TT_BATCH_SIZE = 4_096

CFG = BenchConfig(
    ws_path=str(ROOT / "panda_test"),
    links=["panda_link0", "panda_link1", "panda_link2", "panda_link3"],
    n_func=128,
    cp_ranks=(32, 64),
    tt_ranks=((16, 16), (24, 24)),
    eval_points=(30_000, 120_000),
    eval_source="both",
    warmup_passes=3,
    timing_passes=10,
    train_iters_weights=TRAIN_ITERS_WEIGHTS,
    train_iters_cp=TRAIN_ITERS_CP,
    train_iters_tt=TRAIN_ITERS_TT,
    cp_tt_source=CP_TT_SOURCE,
    cp_train_method=CP_METHOD,
    cp_train_lr=CP_LR,
    cp_train_ridge=CP_RIDGE,
    cp_train_batch_size=CP_BATCH_SIZE,
    tt_train_method=TT_METHOD,
    tt_train_lr=TT_LR,
    tt_train_ridge=TT_RIDGE,
    tt_train_batch_size=TT_BATCH_SIZE,
    run_train_weights=RUN_TRAIN_WEIGHTS,
    include_weights_method=INCLUDE_WEIGHTS_METHOD,
    allow_weight_upsample=False,
)

if __name__ == "__main__":
    run_benchmark_suite(CFG)
