from pathlib import Path

import mesh_to_sdf
import numpy as np
import torch

from panda_test.panda_fk import fk_panda_zero_list
from src.core.sdf_creator import SDFTrain
from src.core.train.train_config import CPLinkCfg, CPRobotCfg, TTLinkCfg, TTRobotCfg, TrainConfig, Train_W


ROOT = Path(__file__).resolve().parent
WS_PATH = str(ROOT / "panda_test")


device = "cuda" if torch.cuda.is_available() else "cpu"
trainer = SDFTrain(device=device)
trainer.init_robot_folder(WS_PATH, robot_name="panda")


test_cfg = TrainConfig(
    debug=True,
    links_to_train=[
        "panda_link0",
        "panda_link1",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
        "panda_link7",
    ],
    fk_matrices=fk_panda_zero_list(),
    classic=Train_W(run=True, n_func=8, iters=200, batch_near=1024, batch_rand=64),
    cp_link=CPLinkCfg(run=True, n_func=64, rank=32, iters=1000, ridge=1e-4),
    cp_robot=CPRobotCfg(run=False, n_func=64, rank=32, iters=40, ridge=1e-4, batch_size=65_536),
    tt_link=TTLinkCfg(run=False, n_func=24, ranks=(8, 16), iters=1000, ridge=1e-4, lr=1e-2),
    tt_robot=TTRobotCfg(run=False, n_func=24, ranks=(8, 16, 8), iters=40, ridge=1e-4, lr=1e-2, batch_size=65_536),
)
# [trainer.create_dataset(link_name, robot_name='panda',) for link_name in ['cube', 'sphere', 'letter_C', 'letter_C_rounded', 'parallelepiped']]

trainer.create_model(test_cfg, robot_name="panda")
