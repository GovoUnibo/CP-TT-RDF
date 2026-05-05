from pathlib import Path

import torch

from src.rdf_3Dcp import RDF_3D_CP

ROOT = Path(__file__).resolve().parent
WS_PATH = str(ROOT / "panda_test")
ROBOT_NAME = 'panda_robot'

BASE_LINK_NAMES = [
    'panda_link0',
    'panda_link1',
    'panda_link2',
    'panda_link3',
    'panda_link4',
    'panda_link5',
    'panda_link6',
    'panda_link7',
    'panda_hand',
    'panda_leftfinger',
    'panda_rightfinger',
]

N_FUNC = 8
CP_RANK = 24
TRAIN_ITERS = 100


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rdf = RDF_3D_CP(device=device)
    rdf.init_robot_folder(WS_PATH, robot_name=ROBOT_NAME)
    rdf.train_links(
        link_names=BASE_LINK_NAMES,
        n_func=N_FUNC,
        ranks=CP_RANK,
        iters=TRAIN_ITERS,
        robot_name=ROBOT_NAME,
        debug=False,
    )


if __name__ == '__main__':
    main()
