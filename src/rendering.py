from jaxtyping import Float
from torch import Tensor, arange, ones, round

from .geometry import homogenize_points, project, transform_world2cam


def render_point_cloud(
    vertices: Float[Tensor, "vertex 3"],
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    resolution: tuple[int, int] = (256, 256),
) -> Float[Tensor, "batch height width"]:
    """Create a white canvas with the specified resolution. Then, transform the points
    into camera space, project them onto the image plane, and color the corresponding
    pixels on the canvas black.
    """

    batch_size = extrinsics.shape[0]
    homo_verts = homogenize_points(vertices)

    # print(homo_verts.unsqueeze(0).shape, extrinsics.unsqueeze(1).shape)
    vertices_camera = transform_world2cam(
        homo_verts.unsqueeze(0), extrinsics.unsqueeze(1)
    )
    projected_points = project(vertices_camera, intrinsics.unsqueeze(1))
    canvas = ones((batch_size, resolution[1], resolution[0]), device=vertices.device)

    x_indices = (
        round(projected_points[..., 0] * (resolution[0] - 1))
        .long()
        .clamp(0, resolution[0] - 1)
    )

    y_indices = (
        round(projected_points[..., 1] * (resolution[1] - 1))
        .long()
        .clamp(0, resolution[1] - 1)
    )

    batch_indices = (
        arange(batch_size, device=vertices.device).unsqueeze(1).expand_as(x_indices)
    )

    x_indices_flat = x_indices.flatten()
    y_indices_flat = y_indices.flatten()
    batch_indices_flat = batch_indices.flatten()

    # print(x_indices_flat.shape, y_indices_flat.shape, batch_indices_flat.shape)

    canvas[batch_indices_flat, y_indices_flat, x_indices_flat] = 0

    return canvas
