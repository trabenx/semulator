import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.transform import warp # Keep using skimage warp for convenience

def apply_affine(image, masks, params, rng):
    """Applies affine transform. Operates on float image, uint masks."""
    # (Implementation as before)
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    max_scale = params.get('max_scale_delta', 0.1)
    max_rot = params.get('max_rotation_deg', 10)
    max_shear = params.get('max_shear_deg', 5)
    max_trans = params.get('max_translate_fraction', 0.05)

    scale = 1.0 + rng.uniform(-max_scale, max_scale)
    angle = rng.uniform(-max_rot, max_rot)
    shear_x = rng.uniform(-max_shear, max_shear)
    trans_x = rng.uniform(-max_trans, max_trans) * w
    trans_y = rng.uniform(-max_trans, max_trans) * h

    M_rot_scale = cv2.getRotationMatrix2D(center, angle, scale)
    M_rot_scale[0, 2] += trans_x
    M_rot_scale[1, 2] += trans_y
    shear_rad = np.deg2rad(shear_x)
    M_final = M_rot_scale.copy()
    M_final[0, 1] += np.tan(shear_rad) # Shear X approx

    # Calculate inverse matrix for warp field calculation
    M_inv = cv2.invertAffineTransform(M_final)

    # Apply transformation
    # Use REFLECT border for image, CONSTANT 0 for masks
    warped_image = cv2.warpAffine(image, M_final, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)

    warped_masks = []
    if masks is not None:
        for mask in masks:
             if mask is None:
                  warped_masks.append(None)
                  continue
             warped_mask = cv2.warpAffine(mask, M_final, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
             warped_masks.append(warped_mask)

    # Calculate warp field: displacement = new_coord - original_coord
    # original_coord = M_inv * new_coord
    yy, xx = np.indices((h, w), dtype=np.float32)
    new_coords_homogeneous = np.stack([xx.ravel(), yy.ravel(), np.ones(h*w)], axis=1) # Shape (N, 3)
    original_coords_homogeneous = new_coords_homogeneous @ M_inv.T # Apply inverse transform

    original_x = original_coords_homogeneous[:, 0].reshape(h, w)
    original_y = original_coords_homogeneous[:, 1].reshape(h, w)

    # Displacement = original_coord - new_coord (Check convention: often warp = new - old)
    # Let's use warp = M*old - old. Need forward map from original coords.
    # Alternative: warp = new_coord - inverse_map(new_coord)
    # Displacement dy = yy - original_y, dx = xx - original_x
    warp_dy = yy - original_y
    warp_dx = xx - original_x
    warp_field = np.stack((warp_dy, warp_dx), axis=-1)

    print(f"Applied affine transform: scale={scale:.2f}, angle={angle:.1f}, shear={shear_x:.1f}, trans=({trans_x:.1f}, {trans_y:.1f})")
    return warped_image, warped_masks, warp_field


def apply_elastic(image, masks, params, rng):
    """Applies elastic mesh deformation using skimage.transform.warp."""
    # (Implementation mostly as before)
    h, w = image.shape[:2]
    alpha = params.get('alpha', 50)
    sigma = params.get('sigma', 5)
    grid_scale = params.get('grid_scale', 4)

    # Use numpy RandomState for generating the displacement field
    np_rng = np.random.RandomState(rng.randint(0, 2**32 - 1))

    dh = max(1, h // grid_scale)
    dw = max(1, w // grid_scale)
    # Use np_rng.rand which gives [0, 1), shift to [-1, 1)
    dx_coarse = gaussian_filter( (np_rng.rand(dh, dw) * 2 - 1), sigma, mode="reflect") * alpha
    dy_coarse = gaussian_filter( (np_rng.rand(dh, dw) * 2 - 1), sigma, mode="reflect") * alpha

    # Resize displacement fields smoothly
    map_x = cv2.resize(dx_coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(dy_coarse, (w, h), interpolation=cv2.INTER_LINEAR)

    # Create coordinate mesh for warping (input coords for each output pixel)
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    # Displacement map indicates where *from* to sample for the current pixel
    # map_coordinates expects (coords_dim, H, W)
    indices = np.stack([y_coords + map_y, x_coords + map_x], axis=0)

    # Warp image (order 1 = bilinear)
    # Using map_coordinates directly can be faster for certain interpolations
    # warped_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(h, w)
    # Sticking to skimage.transform.warp for consistency for now
    warped_image = warp(image, indices, order=1, mode='reflect', cval=0, preserve_range=True).astype(image.dtype)


    # Warp masks (order 0 = nearest neighbor)
    warped_masks = []
    if masks is not None:
       for mask in masks:
            if mask is None:
                warped_masks.append(None)
                continue
            # warped_mask = map_coordinates(mask, indices, order=0, mode='constant', cval=0).reshape(h, w).astype(mask.dtype)
            warped_mask = warp(mask, indices, order=0, mode='constant', cval=0, preserve_range=True).astype(mask.dtype)
            warped_masks.append(warped_mask)

    # Warp field = displacement = new_coord - original_coord
    # Here, map_x/map_y represents the displacement dx/dy. Check sign.
    # indices = original_coords + displacement => displacement = indices - original_coords
    # So warp_field = (map_y, map_x)
    warp_field = np.stack((map_y, map_x), axis=-1) # H x W x 2 (dy, dx)

    print(f"Applied elastic deformation: alpha={alpha:.1f}, sigma={sigma:.1f}")
    return warped_image, warped_masks, warp_field
