from jaxtyping import Float
from torch import Tensor, cat, einsum, ones, zeros


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional points into (n+1)-dimensional homogeneous points."""

    return cat((points, ones(points.shape[:-1] + (1,))), dim=-1)


def homogenize_vectors(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional vectors into (n+1)-dimensional homogeneous vectors."""

    return cat((points, zeros(points.shape[:-1] + (1,))), dim=-1)


def transform_rigid(
    xyz: Float[Tensor, "*#batch 4"],
    transform: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Apply a rigid-body transform to homogeneous points or vectors."""

    return einsum("...ij,...j->...i", transform, xyz)


def transform_world2cam(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D world coordinates to homogeneous
    3D camera coordinates.
    """

    world2cam = cam2world.inverse()
    return transform_rigid(xyz, world2cam)


def transform_cam2world(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D camera coordinates to homogeneous
    3D world coordinates.
    """

    return transform_rigid(xyz, cam2world)


def project(
    xyz: Float[Tensor, "*#batch 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch 2"]:
    """Project homogenized 3D points in camera coordinates to pixel coordinates."""

    # projection_matrix = cat((intrinsics, zeros(intrinsics.shape[:-1] + (1,))), dim=-1)
    projection_matrix = homogenize_vectors(intrinsics)
    # print(projection_matrix.shape)
    homo_coords = einsum("...ij,...j->...i", projection_matrix, xyz)
    # homo_coords = transform_rigid(xyz, projection_matrix)
    # print(homo_coords.shape)
    img_coords = homo_coords[..., :2] / homo_coords[..., 2:3]
    return img_coords
