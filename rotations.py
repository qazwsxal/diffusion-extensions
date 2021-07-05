from functools import reduce
from operator import matmul
from typing import Tuple
import torch


def rmat2six(x: torch.Tensor) -> torch.Tensor:
    # Drop last column
    return torch.flatten(x[..., :2, :], -2, -1)


def six2rmat(x: torch.Tensor) -> torch.Tensor:
    a1 = x[..., :3]
    a2 = x[..., 3:6]
    b1 = a1 / a1.norm(dim=-1, keepdim=True)
    b1_a2 = (b1 * a2).sum(dim=-1, keepdim=True)  # Dot product
    b2 = (a2 - b1_a2 * b1)
    b2 = b2 / b2.norm(dim=-1, keepdim=True)
    b3 = torch.cross(b1, b2, dim=-1)
    out = torch.stack((b1, b2, b3), dim=-2)
    return out


def rmat_cosine_dist(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    ''' Calculate the cosine distance between two (batched) rotation matrices

    '''
    # rotation matrices have A^-1 = A_transpose
    # multiplying the inverse of m2 by m1 gives us a combined transform
    # corresponding to roting by m1 and then back by the *opposite* of m2.
    # if m1 == m2, m_comb = identity
    m_comb = m2.transpose(-1, -2) @ m1
    # Batch trace calculation
    tra = torch.einsum('...ii', m_comb)
    # The trace of a rotation matrix = 1+2*cos(a)
    # where a is the angle in the axis-angle representation
    # Cosine dist is defined as 1 - cos(a)
    out = 1 - (tra - 1) / 2
    return out


def arb_axis_amat_gen(point_1, point_2, s_ang, c_ang):
    '''Generates an affine matrix defining a rotation about the axis specified by point_1 and point 2

    `point_1`: Point in 3D space
    `point_2`: Point in 3D space
    `s_ang`: sine of rotation angle
    `c_ang`: cosine of rotation angle
    '''
    mat_ops = []
    # Translate everything to origin
    translate = torch.eye(4, device=point_1.device)
    translate[:3, 3] = -point_1[:3]
    mat_ops.insert(0, translate)
    # Calculate axis of rotation and normalise
    rot_axis = point_1 - point_2
    rot_mat = torch.eye(4).to(point_1)
    rot_mat[:3, :3] = aa_to_rmat(rot_axis, s_ang, c_ang)
    mat_ops.insert(0, rot_mat)
    # Translate everything back from origin
    translate2 = torch.eye(4, device=point_1.device)
    translate2[:3, 3] = point_1[:3]
    mat_ops.insert(0, translate2)
    # Reduce to single matrix.
    mat = reduce(matmul, mat_ops)
    # mat[:3,:3] should be an orthogonal matrix.
    mat = orthogonalise(mat)
    return mat


def aa_to_rmat(rot_axis, ang):
    '''Generates a rotation matrix (3x3) from axis-angle form

        `rot_axis`: Axis to rotate around, defined as vector from origin.
        `ang`: rotation angle
        '''
    # Gnarly axis-angle to matrix rotation thing, check wiki.
    s_ang, c_ang = ang.sin(), ang.cos()
    rot_axis_n = rot_axis / rot_axis.norm(dim=-1, keepdim=True)
    I = torch.eye(3, device=rot_axis_n.device).expand((*rot_axis.shape[:-1], -1, -1))
    cpm = torch.zeros((*rot_axis.shape[:-1], 3, 3), device=rot_axis_n.device)
    cpm[..., 2, 1] = rot_axis_n[..., 0]
    cpm[..., 2, 0] = -rot_axis_n[..., 1]
    cpm[..., 1, 0] = rot_axis_n[..., 2]
    # Skew symmetric matrix
    cpm += -cpm.transpose(-1, -2)
    op = rot_axis_n[..., :3, None] * rot_axis_n[..., :3, None].transpose(-1, -2)
    # Angle is in sin-cosine form, so we don't need to calculate them again
    rot_mat = c_ang[..., None] * I + s_ang[..., None] * cpm + (1 - c_ang[..., None]) * op
    rot_mat_o = orthogonalise(rot_mat)
    return rot_mat_o


# We use atan2 instead of acos here dut to better numerical stability.
# it means we get nicer behaviour around 0 degrees
# More effort to derive sin terms
# but as we're dealing with small angles a lot,
# the tradeoff is worth it.
def rmat_to_aa(r_mat) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Calculates axis and angle of rotation from a rotation matrix.

        returns angles in [0,pi] range.

        `r_mat`: rotation matrix.
        '''
    skew_mat = 0.5 * (r_mat - r_mat.transpose(-1,-2))
    sk_vec = torch.empty((*skew_mat.shape[:-2], 3))
    sk_vec[...,0] = skew_mat[...,2,1]
    sk_vec[...,1] = -skew_mat[..., 2,0]
    sk_vec[...,2] = skew_mat[...,1,0]
    s_angle = sk_vec.norm(dim=-1)
    axis = sk_vec/s_angle
    c_angle =(torch.einsum('...ii', r_mat)-1)/2
    angle = torch.atan2(s_angle, c_angle)
    # At this point, the determined axis and angles might not match the sign of what we want,
    # Check if we can recover the r_mat properly:
    if torch.allclose(aa_to_rmat(axis, angle), r_mat):
        return axis, angle
    else:
        return -axis, angle


def log_rmat(r_mat: torch.Tensor) -> torch.Tensor:
    skew_mat = (r_mat - r_mat.transpose(-1, -2))
    sk_vec = torch.empty((*skew_mat.shape[:-2], 3))
    sk_vec[..., 0] = skew_mat[..., 2, 1]
    sk_vec[..., 1] = -skew_mat[..., 2, 0]
    sk_vec[..., 2] = skew_mat[..., 1, 0]
    s_angle = (sk_vec/2).norm(dim=-1)
    c_angle = (torch.einsum('...ii', r_mat) - 1) / 2
    angle = torch.atan2(s_angle, c_angle)
    scale = (angle / (2 * s_angle))
    # if s_angle = 0, i.e. rotation by 0 or pi, we get NaNs
    # by definition, scale values are 0 if rotating by 0.
    scale[angle==0] = 0
    # This also breaks down if rotating by pi, but idk how to fix that.
    log_r_mat = scale[..., None, None] * skew_mat

    return log_r_mat


def rmat_dist(input, target):
    '''Calculates the geodesic distance between two (batched) rotation matrices.

    '''
    return log_rmat(input.transpose(-1,-2) @ target).norm(p='fro') # Frobenius nrm


def so3_lerp(rot_a: torch.Tensor, rot_b: torch.Tensor, weight: torch.Tensor):
    ''' Weighted interpolation between rot_a and rot_b

    '''
    # Treat rot_b = rot_a @ rot_c
    # rot_a^-1 @ rot_a = I
    # rot_a^-1 @ rot_b = rot_a^-1 @ rot_a @ rot_c = I @ rot_c
    rot_c = rot_a.transpose(-1, -2) @ rot_b
    axis, angle = rmat_to_aa(rot_c)
    # once we have axis-angle forms, determine intermediate angles.
    i_angle = weight * angle
    rot_c_i = aa_to_rmat(axis, i_angle)
    return rot_a @ rot_c_i


def orthogonalise(mat):
    """Orthogonalise rotation/affine matrices

    Ideally, 3D rotation matrices should be orthogonal,
    however during creation, floating point errors can build up.
    We QR decompose our matrix as in the ideal case S is a diagonal matrix of 1s
    We then round the values of S to [-1, 0, +1],
    making U @ S_rounded @ V.T an orthonormal matrix close to the original.
    """
    # TODO fix this so it wo
    orth_mat = mat.clone()
    u, s, v = torch.svd(mat[..., :3, :3])
    orth_mat[..., :3, :3] = u @ torch.diag_embed(s.round()) @ v.transpose(-1,-2)
    return orth_mat


if __name__ == "__main__":
    vals = torch.randn((3, 4, 6))
    mats = six2rmat(vals)
    valback = rmat2six(mats)
    m1 = mats[:, :2]
    m2 = mats[:, 2:]
    res = rmat_cosine_dist(m1, m2)
    res2 = rmat_cosine_dist(m1, m1)
    res3 = rmat_cosine_dist(m2, m2)
    weight = 0.2
    out = so3_lerp(m1[0, 0], m2[0, 0], weight)
    log_m1 = log_rmat(m1[0,0])
    log_rmat(torch.eye(3))