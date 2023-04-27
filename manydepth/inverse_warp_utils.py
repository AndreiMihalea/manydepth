import torch


pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(
        b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def get_factor(depth: torch.tensor, intrinsic):
    global pixel_coords
    """
    @param disp: depth map, [B, 1, H, W]
    :returns depth factor
    """
    batch_size, _, height, width = depth.shape
    CAM_HEIGHT = 1.65  # 1.54 for UPB

    # construct intrinsic camera matrix
    intrinsic = torch.tensor(intrinsic).repeat(batch_size, 1, 1)

    # get camera coordinates
    cam_coords = pixel2cam(depth.squeeze(1), intrinsic.inverse())

    # get some samples from the ground, center of the image
    samples = cam_coords[:, 1, height - 10:height, width // 2 - 50:width // 2 + 50]
    samples = samples.reshape(samples.shape[0], -1)

    # get the median
    median = samples.median(1)[0]

    # get depth factor
    factor = CAM_HEIGHT / median
    pixel_coords = None
    return factor.reshape(factor.shape, 1, 1, 1)