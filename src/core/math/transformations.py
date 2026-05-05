import torch
import numpy as np
from typing import Union, List

def rotation_matrix_to_quaternion(rotMatrix, device='cpu'):
    # Assicurati che la matrice di rotazione sia sul dispositivo corretto
    if not isinstance(rotMatrix, torch.Tensor):
        rotMatrix = torch.tensor(rotMatrix, device=device)

    # Estrai i componenti dalla matrice di rotazione
    r11, r12, r13 = rotMatrix[0, 0], rotMatrix[0, 1], rotMatrix[0, 2]
    r21, r22, r23 = rotMatrix[1, 0], rotMatrix[1, 1], rotMatrix[1, 2]
    r31, r32, r33 = rotMatrix[2, 0], rotMatrix[2, 1], rotMatrix[2, 2]

    # Valori costanti
    one = torch.tensor(1.0, device=device)
    zero = torch.tensor(0.0, device=device)
    four = torch.tensor(4.0, device=device)

    # Calcola i valori iniziali dei quaternioni
    q = torch.zeros(4, device=device)
    q[0] = torch.clamp((r11 + r22 + r33 + one) / four, min=zero).sqrt()  # w
    q[1] = torch.clamp((r11 - r22 - r33 + one) / four, min=zero).sqrt()  # x
    q[2] = torch.clamp((-r11 + r22 - r33 + one) / four, min=zero).sqrt()  # y
    q[3] = torch.clamp((-r11 - r22 + r33 + one) / four, min=zero).sqrt()  # z

    # Determina il quaternione dominante e aggiorna i segni
    if q[0] >= q[1] and q[0] >= q[2] and q[0] >= q[3]:
        q[0], q[1], q[2], q[3] = q[0], q[1] * torch.sign(r32 - r23), q[2] * torch.sign(r13 - r31), q[3] * torch.sign(r21 - r12)
    elif q[1] >= q[0] and q[1] >= q[2] and q[1] >= q[3]:
        q[0], q[1], q[2], q[3] = q[0] * torch.sign(r32 - r23), q[1], q[2] * torch.sign(r21 + r12), q[3] * torch.sign(r13 + r31)
    elif q[2] >= q[0] and q[2] >= q[1] and q[2] >= q[3]:
        q[0], q[1], q[2], q[3] = q[0] * torch.sign(r13 - r31), q[1] * torch.sign(r21 + r12), q[2], q[3] * torch.sign(r32 + r23)
    elif q[3] >= q[0] and q[3] >= q[1] and q[3] >= q[2]:
        q[0], q[1], q[2], q[3] = q[0] * torch.sign(r21 - r12), q[1] * torch.sign(r31 + r13), q[2] * torch.sign(r32 + r23), q[3]
    else:
        raise ValueError("Coding error")

    # Normalizza il quaternione
    q /= torch.norm(q)

    return q

def rpy_to_quaterion(roll, pitch, yaw):
    '''
        q0 = w , q1 = x, q2 = y, q3 = z
    '''
    q = np.zeros(4)
    q[1] = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2) # x
    q[2] = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) # y
    q[3] = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) # z
    q[0] = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2) # w
    return q

def matrix_to_pos_quat(matrix, device='cpu'):
    '''
    Extracts the position and quaternion from a 4x4 matrix
    :param matrix: 4x4 matrix
    :return: position, quaternion
    '''
    return matrix[:3, 3], rotation_matrix_to_quaternion(matrix[:3, :3], device=device)


def homogeneous_matrix(x, y, z, psi, theta, phi, dtype=torch.float64, device='cpu'):

    # Convert angles to the specified dtype and device
    phi = phi.to(dtype).to(device)
    theta = theta.to(dtype).to(device)
    psi = psi.to(dtype).to(device)
    
    # Precompute cosines and sines of angles
    cos_phi = torch.cos(phi).to(dtype)
    sin_phi = torch.sin(phi).to(dtype)
    cos_theta = torch.cos(theta).to(dtype)
    sin_theta = torch.sin(theta).to(dtype)
    cos_psi = torch.cos(psi).to(dtype)
    sin_psi = torch.sin(psi).to(dtype)
    
    # Create the homogeneous transformation matrix on the correct device
    h_matrix = torch.tensor([
        [cos_phi * cos_theta, 
         cos_phi * sin_theta * sin_psi - sin_phi * cos_psi, 
         cos_phi * sin_theta * cos_psi + sin_phi * sin_psi, 
         x],

        [sin_phi * cos_theta, 
         sin_phi * sin_theta * sin_psi + cos_phi * cos_psi,
         sin_phi * sin_theta * cos_psi - cos_phi * sin_psi, 
         y],

        [-sin_theta,
         cos_theta * sin_psi,
         cos_theta * cos_psi,
         z],

        [torch.tensor(0.0, dtype=dtype, device=device), 
         torch.tensor(0.0, dtype=dtype, device=device),
         torch.tensor(0.0, dtype=dtype, device=device),
         torch.tensor(1.0, dtype=dtype, device=device)]
    ], dtype=dtype, device=device)
    
    return h_matrix


def direct_transform_points(matrix, points, device='cuda', dtype=torch.float32)-> Union[torch.Tensor, np.ndarray]:
    '''
        matrix: (4, 4)
        points: (N, 3)
        return: (N, 3)'''
    original_type = type(points)

    # Converti input a tensori se necessario
    if isinstance(points, list):
        points = torch.tensor(points, dtype=dtype, device=device)
    elif isinstance(points, np.ndarray):
        points = torch.tensor(points, dtype=dtype, device=device)

    if isinstance(matrix, list):
        matrix = torch.tensor(matrix, dtype=dtype, device=device)
    elif isinstance(matrix, np.ndarray):
        matrix = torch.tensor(matrix, dtype=dtype, device=device)

    # Assicurati che siano sul device corretto
    points = points.to(device)
    matrix = matrix.to(device)

    # Aggiungi dimensione batch se la matrice è singola
    if matrix.ndim == 2:  # (4, 4)
        matrix = matrix.unsqueeze(0)  # (1, 4, 4)

    # Aggiungi la componente omogenea: (N, 4)
    ones = torch.ones((points.shape[0], 1), device=device, dtype=dtype)
    homogeneous_points = torch.cat([points, ones], dim=1)  # (N, 4)
    homogeneous_points = homogeneous_points.unsqueeze(0)  # (1, N, 4)

    # Trasformazione: (1, 4, 4) x (1, N, 4)^T -> (1, N, 4)
    transformed_points_homogeneous = torch.matmul(matrix, homogeneous_points.transpose(-1, -2)).transpose(-1, -2)  # (1, N, 4)

    # Estrai (N, 3)
    transformed_points = transformed_points_homogeneous[0, :, :3]

    # Converti al tipo originale se necessario
    if original_type == torch.Tensor:
        return transformed_points
    elif original_type in [list, np.ndarray]:
        transformed_points = transformed_points.cpu().numpy()
        if original_type == list:
            transformed_points = transformed_points.tolist()
        return transformed_points


def direct_transform_points_batch(matrix, points, device='cuda', dtype=torch.float32) -> Union[torch.Tensor, np.ndarray]:
    '''
        matrix: (B, 4, 4)
        points: (B, N, 3)
        return: (B, N, 3)
    '''
    original_type = type(points)

    if isinstance(points, list):
        points = torch.tensor(points, dtype=dtype, device=device)
    elif isinstance(points, np.ndarray):
        points = torch.tensor(points, dtype=dtype, device=device)

    if isinstance(matrix, list):
        matrix = torch.tensor(matrix, dtype=dtype, device=device)
    elif isinstance(matrix, np.ndarray):
        matrix = torch.tensor(matrix, dtype=dtype, device=device)

    # Se sono già tensori PyTorch, assicurati che siano nel device corretto
    if isinstance(points, torch.Tensor):
        points = points.to(device=device, dtype=dtype)
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.to(device=device, dtype=dtype)

    if matrix.ndim != 3 or matrix.shape[1:] != (4, 4):
        raise ValueError("matrix must be of shape (B, 4, 4)")
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("points must be of shape (B, N, 3)")

    B, N, _ = points.shape
    ones = torch.ones((B, N, 1), device=device, dtype=dtype)
    homogeneous_points = torch.cat([points, ones], dim=2)  # (B, N, 4)

    transformed = torch.matmul(homogeneous_points, matrix.transpose(1, 2))  # (B, N, 4)
    transformed_points = transformed[:, :, :3]

    if original_type == torch.Tensor:
        return transformed_points
    elif original_type in [list, np.ndarray]:
        transformed_points = transformed_points.cpu().numpy()
        if original_type == list:
            transformed_points = transformed_points.tolist()
        return transformed_points

def vector6_to_homogeneus_batch(vector6:torch.Tensor, device=None, dtype=torch.float32):
    if not vector6.dim() == 2 or not vector6.size(1) == 6:
        raise ValueError("Input vector6 must have shape (batch, 6)")
        
    # Ensure input shape is (batch, 6)
    batch_input = vector6.squeeze(-1)  # Remove the singleton dimension if present
    batch_size = batch_input.size(0)

    # Split the input into x, y, z, roll, pitch, yaw
    x, y, z = batch_input[:, 0], batch_input[:, 1], batch_input[:, 2]
    roll, pitch, yaw = batch_input[:, 3], batch_input[:, 4], batch_input[:, 5]

   
    # Compute trigonometric values
    cos_r, sin_r = torch.cos(roll), torch.sin(roll)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

    # Rotation matrices for roll (Rx), pitch (Ry), and yaw (Rz)
    
    R_x = torch.stack([
        torch.stack([torch.ones(batch_size, device=device, dtype=dtype), torch.zeros(batch_size, device=device, dtype=dtype), torch.zeros(batch_size, device=device, dtype=dtype)], dim=1),
        torch.stack([torch.zeros(batch_size, device=device, dtype=dtype), cos_r, -sin_r], dim=1),
        torch.stack([torch.zeros(batch_size, device=device, dtype=dtype), sin_r, cos_r], dim=1)
    ], dim=2)  # (batch, 3, 3)

    R_y = torch.stack([
        torch.stack([cos_p, torch.zeros(batch_size, device=device, dtype=dtype), sin_p], dim=1),
        torch.stack([torch.zeros(batch_size, device=device, dtype=dtype), torch.ones(batch_size, device=device, dtype=dtype), torch.zeros(batch_size, device=device, dtype=dtype)], dim=1),
        torch.stack([-sin_p, torch.zeros(batch_size, device=device, dtype=dtype), cos_p], dim=1)
    ], dim=2)  # (batch, 3, 3)

    R_z = torch.stack([
        torch.stack([cos_y, -sin_y, torch.zeros(batch_size, device=device, dtype=dtype)], dim=1),
        torch.stack([sin_y, cos_y, torch.zeros(batch_size, device=device, dtype=dtype)], dim=1),
        torch.stack([torch.zeros(batch_size, device=device, dtype=dtype), torch.zeros(batch_size, device=device, dtype=dtype), torch.ones(batch_size, device=device, dtype=dtype)], dim=1)
    ], dim=2)  # (batch, 3, 3)

    # Full rotation matrix: R = Rz * Ry * Rx
    R = torch.bmm(torch.bmm(R_z, R_y), R_x)  # (batch, 3, 3)

    # Translation vector
    T = torch.stack([x, y, z], dim=1).unsqueeze(2)  # (batch, 3, 1)

    # Combine rotation and translation into homogeneous transformation matrix
    RT = torch.cat([R, T], dim=2)  # (batch, 3, 4)

    # Add the bottom row [0, 0, 0, 1] to make it 4x4
    bottom_row = torch.tensor([0, 0, 0, 1], dtype=dtype, device=device)
    bottom_row = bottom_row.view(1, 1, 4).repeat(batch_size, 1, 1)  # (batch, 1, 4)

    homogeneous_matrices = torch.cat([RT, bottom_row], dim=1)  # (batch, 4, 4)
    return homogeneous_matrices



def invert_homogeneous_batch(T: torch.Tensor) -> torch.Tensor:
        """
        Invert a batch of homogeneous transforms.
        T: (..., 4, 4) tensor
        Returns: (..., 4, 4) tensor of inverses
        """
        R = T[..., :3, :3]              # rotation part (..., 3, 3)
        t = T[..., :3, 3:]              # translation (..., 3, 1)

        R_inv = R.transpose(-1, -2)     # R^T
        t_inv = -R_inv @ t              # -R^T * t

        T_inv = torch.zeros_like(T)
        T_inv[..., :3, :3] = R_inv
        T_inv[..., :3, 3:] = t_inv
        T_inv[..., 3, 3] = 1.0

        return T_inv


def forward_pc_batch(robot_pc: torch.Tensor, fk_batch_tensor: torch.Tensor) -> torch.Tensor:
    """
    Trasforma tutte le pointcloud dei link usando la FK di un batch di configurazioni.

    Args:
        robot_pc: [L, N_point, 3]  punti dei link in locale
        fk_batch_tensor: [C, L, 4, 4]  batch di matrici di trasformazione (una per ogni config e link)

    Returns:
        pc_transformed: [C, L, N_point, 3]  punti trasformati nello spazio mondo
    """
    device, dtype = robot_pc.device, robot_pc.dtype
    L, N, _ = robot_pc.shape
    C = fk_batch_tensor.shape[0]

    # punti omogenei: (1, L, N, 4)
    ones = torch.ones((1, L, N, 1), device=device, dtype=dtype)
    pc_hom = torch.cat([robot_pc.unsqueeze(0).expand(C, -1, -1, -1), ones.expand(C, -1, -1, -1)], dim=-1)  
    # shape: (C, L, N, 4)

    # trasformazioni: (C, L, N, 4) x (C, L, 4, 4) -> (C, L, N, 4)
    pc_transformed = torch.matmul(pc_hom, fk_batch_tensor.transpose(-1, -2))  

    return pc_transformed[..., :3]  # (C, L, N, 3)

def vector6_to_homogeneus_batchZYX(vector6: torch.Tensor, device=None, dtype=torch.float32):
    if vector6.ndim != 2 or vector6.shape[1] != 6:
        raise ValueError("Input vector6 must have shape (batch, 6)")

    device = vector6.device if device is None else device
    vector6 = vector6.to(device=device, dtype=dtype)
    batch_size = vector6.shape[0]

    x, y, z, roll, pitch, yaw = vector6.T

    cos_r, sin_r = torch.cos(roll), torch.sin(roll)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

    # Rx
    R_x = torch.stack([
        torch.stack([torch.ones_like(cos_r), torch.zeros_like(cos_r), torch.zeros_like(cos_r)], dim=1),
        torch.stack([torch.zeros_like(cos_r), cos_r, -sin_r], dim=1),
        torch.stack([torch.zeros_like(cos_r), sin_r,  cos_r], dim=1),
    ], dim=2)

    # Ry
    R_y = torch.stack([
        torch.stack([cos_p, torch.zeros_like(cos_p), -sin_p], dim=1),
        torch.stack([torch.zeros_like(cos_p), torch.ones_like(cos_p), torch.zeros_like(cos_p)], dim=1),
        torch.stack([sin_p, torch.zeros_like(cos_p), cos_p], dim=1),
    ], dim=2)

    # Rz (rotazione intorno a Z)
    R_z = torch.stack([
        torch.stack([cos_y, sin_y, torch.zeros_like(cos_y)], dim=1),
        torch.stack([-sin_y, cos_y, torch.zeros_like(cos_y)], dim=1),
        torch.stack([torch.zeros_like(cos_y), torch.zeros_like(cos_y), torch.ones_like(cos_y)], dim=1),
    ], dim=2)

    R = torch.bmm(torch.bmm(R_z, R_y), R_x)  # convenzione ZYX


    # Traslazione
    T = torch.stack([x, y, z], dim=1).unsqueeze(2)

    # Componi omogenea
    RT = torch.cat([R, T], dim=2)  # (batch, 3, 4)
    bottom = torch.tensor([0, 0, 0, 1], dtype=dtype, device=device).view(1,1,4).repeat(batch_size,1,1)
    H = torch.cat([RT, bottom], dim=1)  # (batch, 4, 4)

    return H