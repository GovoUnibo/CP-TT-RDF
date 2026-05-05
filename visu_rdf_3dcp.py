from pathlib import Path

import numpy as np
import torch

from panda_test.panda_fk import fk_panda_two_robots
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
NAMESPACES = ['p1_', 'p2_']
LINK_NAMES = [ns + ln for ns in NAMESPACES for ln in BASE_LINK_NAMES]

Q_READY = np.array([0.0, -0.7, -0.0, -2.3, -0.0, 1.57, 0.0], dtype=np.float32)
THETA = -np.pi / 10.0
Y_OFFSET = 0.4

RUN_GD_VISU = False
GD_ITERS = 10
GD_EPS = 1e-3

MESH_COLOR = '#C7C7C7'
MESH_OPACITY = 1.0
RDF_COLOR = '#C7C7C7'
RDF_OPACITY = 1.0


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    rdf = RDF_3D_CP(device=device)
    rdf.init_robot_folder(WS_PATH, robot_name=ROBOT_NAME)

    for ns in NAMESPACES:
        rdf.add_models(link_names=BASE_LINK_NAMES, namespace=ns, robot_name=ROBOT_NAME)
        rdf.add_mesh(link_names=BASE_LINK_NAMES, namespace=ns, robot_name=ROBOT_NAME)

    rdf.set_ordered_batch_params(link_names=LINK_NAMES)

    points = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
        ],
        dtype=dtype,
        device=device,
    )
    batch_points = points.unsqueeze(0).repeat(len(LINK_NAMES), 1, 1)

    sdf, grad, _ = rdf.inference_link_batch(points=batch_points, get_grad=True, get_min=True)
    print('SDF-CP BATCH LINK')
    print(sdf)
    print('GRAD-CP BATCH LINK')
    print(grad)

    rz = np.array(
        [
            [np.cos(THETA), -np.sin(THETA), 0.0],
            [np.sin(THETA), np.cos(THETA), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    base_a = np.eye(4, dtype=np.float32)
    base_a[:3, :3] = rz
    base_a[:3, 3] = np.array([0.0, -Y_OFFSET / 2.0, 0.0], dtype=np.float32)
    base_b = np.eye(4, dtype=np.float32)
    base_b[:3, :3] = rz
    base_b[:3, 3] = np.array([0.0, +Y_OFFSET / 2.0, 0.0], dtype=np.float32)

    fk_two = fk_panda_two_robots(
        q_a=Q_READY,
        q_b=Q_READY,
        base_a=base_a,
        base_b=base_b,
        include_gripper=True,
    )
    if fk_two.shape[1] != len(BASE_LINK_NAMES):
        raise ValueError(
            f'FK returned {fk_two.shape[1]} links, expected {len(BASE_LINK_NAMES)} '
            f'for {BASE_LINK_NAMES}.'
        )

    fw_dict = {}
    for idx_ns, ns in enumerate(NAMESPACES):
        for i, base_name in enumerate(BASE_LINK_NAMES):
            fw_dict[ns + base_name] = torch.as_tensor(fk_two[idx_ns, i], device=device, dtype=rdf.dtype)

    mesh_links = [name for name in LINK_NAMES if name.startswith('p1_')]
    rdf_links = [name for name in LINK_NAMES if name.startswith('p2_')]

    if RUN_GD_VISU:
        points_for_gd = (
            torch.tensor([[1.0, 0.0, 0.5]], dtype=dtype, device=device)
            .unsqueeze(0)
            .repeat(len(LINK_NAMES), 1, 1)
        )
        rdf.visualize_gradient_descent(
            initial_points=points_for_gd,
            forward_dict=fw_dict,
            num_of_iteration=GD_ITERS,
            epsilon=GD_EPS,
            mesh_link_names=mesh_links,
            rdf_link_names=rdf_links,
            mesh_color=MESH_COLOR,
            mesh_opacity=MESH_OPACITY,
            rdf_color=RDF_COLOR,
            rdf_opacity=RDF_OPACITY,
        )

    rdf.visualize_scene(
        forward_as_dict=fw_dict,
        links_as_mesh=True,
        mesh_link_names=mesh_links,
        links_as_rdf=True,
        rdf_link_names=rdf_links,
        mesh_color=MESH_COLOR,
        mesh_opacity=MESH_OPACITY,
        rdf_color=RDF_COLOR,
        rdf_opacity=RDF_OPACITY,
    )


if __name__ == '__main__':
    main()
