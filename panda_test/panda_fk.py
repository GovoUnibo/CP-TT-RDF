import numpy as np


def _as_float_array(x, shape, dtype):
    arr = np.asarray(x, dtype=dtype)
    return arr.reshape(shape)


def _inv_transform(T):
    R = T[:3, :3]
    p = T[:3, 3]
    inv = np.eye(4, dtype=T.dtype)
    inv[:3, :3] = R.T
    inv[:3, 3] = -(R.T @ p)
    return inv


def _rotz(theta, dtype):
    theta64 = np.asarray(theta, dtype=np.float64)
    c = np.cos(theta64)
    s = np.sin(theta64)
    R = np.array(
        [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return R.astype(dtype, copy=False)


def _trans(x, y, z, dtype):
    T = np.eye(4, dtype=dtype)
    T[:3, 3] = np.array([x, y, z], dtype=dtype)
    return T


_PANDA_REL_ZERO_CACHE = {}


def _panda_rel_zero(dtype=np.float32):
    key = np.dtype(dtype).str
    cached = _PANDA_REL_ZERO_CACHE.get(key)
    if cached is not None:
        return cached

    abs_zero = [np.asarray(T, dtype=dtype) for T in fk_panda_zero_list()]
    rel = [np.eye(4, dtype=dtype)]
    for i in range(1, len(abs_zero)):
        rel.append(_inv_transform(abs_zero[i - 1]) @ abs_zero[i])

    _PANDA_REL_ZERO_CACHE[key] = rel
    return rel


def fk_panda_zero_list(include_gripper=False, finger_q=0.0, dtype=np.float32):
    T_list = [

        # panda_link0
        np.array([[1., 0., 0., 0.],
                  [0., 1., 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]], dtype=np.float32),

        # panda_link1
        np.array([[1.   , 0.   , 0.   , 0.   ],
                  [0.   , 1.   , 0.   , 0.   ],
                  [0.   , 0.   , 1.   , 0.333],
                  [0.   , 0.   , 0.   , 1.   ]], dtype=np.float32),

        # panda_link2
        np.array([[ 1.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
                  [ 0.000000e+00, -4.371139e-08,  1.000000e+00,  0.000000e+00],
                  [ 0.000000e+00, -1.000000e+00, -4.371139e-08,  3.330000e-01],
                  [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]],
                 dtype=np.float32),

        # panda_link3
        np.array([[1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
                  [0.0000000e+00, 1.0000000e+00, 0.0000000e+00, 1.3812799e-08],
                  [0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 6.4900005e-01],
                  [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]],
                 dtype=np.float32),

        # panda_link4
        np.array([[ 1.0000000e+00,  0.0000000e+00,  0.0000000e+00,  8.2500003e-02],
                  [ 0.0000000e+00, -4.3711388e-08, -1.0000000e+00,  1.3812799e-08],
                  [ 0.0000000e+00,  1.0000000e+00, -4.3711388e-08,  6.4900005e-01],
                  [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
                 dtype=np.float32),

        # panda_link5
        np.array([[ 1.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
                  [ 0.0000000e+00,  1.0000000e+00,  0.0000000e+00, -2.9723735e-09],
                  [ 0.0000000e+00,  0.0000000e+00,  1.0000000e+00,  1.0330000e+00],
                  [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
                 dtype=np.float32),

        # panda_link6
        np.array([[ 1.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
                  [ 0.0000000e+00, -4.3711388e-08, -1.0000000e+00, -2.9723735e-09],
                  [ 0.0000000e+00,  1.0000000e+00, -4.3711388e-08,  1.0330000e+00],
                  [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
                 dtype=np.float32),

        # panda_link7
        np.array([[ 1.0000000e+00,  0.0000000e+00,  0.0000000e+00,  8.8000000e-02],
                  [ 0.0000000e+00, -1.0000000e+00,  8.7422777e-08, -2.9723735e-09],
                  [ 0.0000000e+00, -8.7422777e-08, -1.0000000e+00,  1.0330000e+00],
                  [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
                 dtype=np.float32),
    ]

    T_list = [np.asarray(T, dtype=dtype) for T in T_list]

    if not include_gripper:
        return T_list

    return _append_gripper_fk(T_list=T_list, finger_q=finger_q, dtype=dtype)


def _append_gripper_fk(T_list, finger_q=0.0, dtype=np.float32):
    finger_q = float(finger_q)
    T7 = T_list[-1]

    # URDF chain:
    # link7 -> link8:    xyz=(0, 0, 0.107), rpy=(0, 0, 0)
    # link8 -> hand:     xyz=(0, 0, 0),     rpy=(0, 0, -0.707)
    # hand  -> fingers:  xyz=(0, +/-q, 0.0584), rpy=(0, 0, 0)
    T_link8 = T7 @ _trans(0.0, 0.0, 0.107, dtype=dtype)
    T_hand = T_link8 @ _rotz(-0.707, dtype=dtype)
    T_leftfinger = T_hand @ _trans(0.0, +finger_q, 0.0584, dtype=dtype)
    T_rightfinger = T_hand @ _trans(0.0, -finger_q, 0.0584, dtype=dtype)

    return T_list + [T_hand, T_leftfinger, T_rightfinger]


def fk_panda(q=None, base=None, dtype=np.float32, include_gripper=False, finger_q=0.0):
    """
    Forward kinematics del Franka Panda (panda_link0..panda_link7).

    - q: array-like shape (7,), valori dei giunti [rad]. Se None -> tutti zero.
    - base: trasformazione 4x4 della base (panda_link0) nel mondo. Se None -> identità.

    Ritorna:
    - include_gripper=False: lista di 8 matrici 4x4 (link0..link7)
    - include_gripper=True : lista di 11 matrici 4x4
      (link0..link7, hand, leftfinger, rightfinger)
    """
    if q is None:
        q = np.zeros(7, dtype=dtype)
    q = _as_float_array(q, (7,), dtype=dtype)

    if base is None:
        T = np.eye(4, dtype=dtype)
    else:
        T = _as_float_array(base, (4, 4), dtype=dtype)

    rel0 = _panda_rel_zero(dtype=dtype)

    T_list = [T.copy()]
    for i in range(1, 8):
        T = T @ rel0[i] @ _rotz(q[i - 1], dtype=dtype)
        T_list.append(T)

    if not include_gripper:
        return T_list

    return _append_gripper_fk(T_list=T_list, finger_q=finger_q, dtype=dtype)


def fk_panda_two_robots(
    offset=np.array([0.6, 0.0, 0.0], dtype=np.float32),
    q_a=None,
    q_b=None,
    base_a=None,
    base_b=None,
    rot_b=None,
    include_gripper=False,
    finger_q_a=0.0,
    finger_q_b=None,
    dtype=np.float32,
):
    """
    Restituisce le pose FK per due robot identici:
      - robot A: fk_panda(q_a, base_a)
      - robot B: fk_panda(q_b, base_b)

    Per compatibilità con la versione precedente:
      - se q_a/q_b sono None -> giunti a zero
      - se base_b è None -> base del robot B = base_a * T(offset, rot_b)
        (prima era solo una traslazione 'offset')

    Ritorna np.ndarray shape (2, L, 4, 4)
    """
    if base_a is None:
        base_a = np.eye(4, dtype=dtype)
    else:
        base_a = _as_float_array(base_a, (4, 4), dtype=dtype)

    if q_a is None:
        q_a = np.zeros(7, dtype=dtype)
    if q_b is None:
        q_b = q_a

    if finger_q_b is None:
        finger_q_b = finger_q_a

    robot_a = np.stack(
        fk_panda(
            q_a,
            base=base_a,
            dtype=dtype,
            include_gripper=include_gripper,
            finger_q=finger_q_a,
        ),
        axis=0,
    )

    if base_b is None:
        T_ab = np.eye(4, dtype=dtype)
        if rot_b is not None:
            T_ab[:3, :3] = _as_float_array(rot_b, (3, 3), dtype=dtype)
        if offset is not None:
            T_ab[:3, 3] = _as_float_array(offset, (3,), dtype=dtype)
        base_b = base_a @ T_ab
    else:
        base_b = _as_float_array(base_b, (4, 4), dtype=dtype)

    robot_b = np.stack(
        fk_panda(
            q_b,
            base=base_b,
            dtype=dtype,
            include_gripper=include_gripper,
            finger_q=finger_q_b,
        ),
        axis=0,
    )

    return np.stack([robot_a, robot_b], axis=0)
