import numpy as np
import cv2
from scipy.ndimage import gaussian_filter # For apply_elastic
from skimage.transform import warp        # For apply_elastic
import logging

TEMP_MASK_PAD_VALUE = 2

# DEBUG
from pathlib import Path # Import Path
import imageio # Import imageio for saving debug images
DEBUG_GEOMETRIC_ARTIFACT_PATH = Path("./debug_geometric_inputs")
#DEBUG_GEOMETRIC_ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
DEBUG_CALL_COUNT = {} # To give unique filenames for multiple calls
def save_debug_mask(mask_array, function_name, list_index, step_name, call_count, logger):
    """Helper to save a mask for debugging."""
    return  # Currently disabled. remove thie line to re-enable
    if mask_array is None:
        logger.debug(f"  Debug Save: Mask item {list_index} in {function_name} at '{step_name}' is None.")
        return
    try:
        # Ensure uint8 for saving as PNG
        if mask_array.dtype != np.uint8:
            save_mask = (mask_array > 0).astype(np.uint8) * 255
        else:
            save_mask = (mask_array > 0).astype(np.uint8) * 255 # Ensure 0 or 255

        filename = f"{function_name}_call{call_count}_mask{list_index}_{step_name}.png"
        filepath = DEBUG_GEOMETRIC_ARTIFACT_PATH / filename
        imageio.imwrite(filepath, save_mask)
        logger.debug(f"  Debug Save: Saved '{filepath}' (sum: {mask_array.sum()}, shape: {mask_array.shape}, dtype: {mask_array.dtype})")
    except Exception as e:
        logger.error(f"  Debug Save: Error saving mask {list_index} in {function_name} at '{step_name}': {e}", exc_info=True)

# -----

def apply_affine(image, input_masks_list, params, rng, logger):
    """
    Applies affine transform to an image and a list of masks.
    All operations and outputs are on the (potentially oversized) input dimensions.
    """
    # DEBUG
    global DEBUG_CALL_COUNT 
    call_key = f"affine_{params.get('name_suffix', 'step')}" # Distinguish calls if multiple affine steps
    DEBUG_CALL_COUNT[call_key] = DEBUG_CALL_COUNT.get(call_key, 0) + 1
    current_call_count = DEBUG_CALL_COUNT[call_key]
    logger.info(f"--- apply_affine Call #{current_call_count} ---")
    if input_masks_list:
        for i, m_in in enumerate(input_masks_list):
            save_debug_mask(m_in, "apply_affine", i, "INPUT", current_call_count, logger)

    # ----
    logger.info(f"--- apply_affine CALLED --- Params: {params}")
    if image is None:
        logger.warning("apply_affine received None for image. Skipping.")
        return None, [None] * len(input_masks_list) if input_masks_list else [], None

    h_canvas, w_canvas = image.shape[:2] # Dimensions of the input (oversized) canvas
    center_canvas_x, center_canvas_y = w_canvas / 2.0, h_canvas / 2.0

    # Get transformation parameters
    max_scale_delta = params.get('max_scale_delta', 0.0)
    max_rot_deg = params.get('max_rotation_deg', 0.0)
    max_shear_deg = params.get('max_shear_deg', 0.0)
    # Translation fraction should be relative to the *visible* part, not the whole oversized canvas,
    # otherwise, a small fraction can lead to huge pixel shifts.
    # If original dimensions (h, w from generator) are not passed, we estimate or use smaller fractions.
    # For simplicity, assume params are tuned for typical image sizes.
    # Or, pass original h,w in params under a key like '_original_dims'
    h_orig, w_orig = params.get('_original_dims', (h_canvas, w_canvas)) # Fallback to canvas size
    max_trans_frac = params.get('max_translate_fraction', 0.01) # Use the direct fraction if available


    # Generate random values for this instance
    scale = 1.0 + rng.uniform(-max_scale_delta, max_scale_delta)
    angle_deg = rng.uniform(-max_rot_deg, max_rot_deg)
    shear_x_deg = rng.uniform(-max_shear_deg, max_shear_deg) # Shear along x-axis

    # Absolute translation amounts
    # If max_translate_fraction was used in config, it implies relative to original image, not oversized.
    # The params from config randomization step should already have single values.
    trans_x = rng.uniform(-max_trans_frac, max_trans_frac) * w_canvas
    trans_y = rng.uniform(-max_trans_frac, max_trans_frac) * h_canvas

    logger.info(f"apply_affine - Canvas: {w_canvas}x{h_canvas}, Scale:{scale:.3f}, Angle:{angle_deg:.1f}, ShearX_deg:{shear_x_deg:.1f}, Trans:({trans_x:.1f},{trans_y:.1f})")

    center_tuple = (center_canvas_x, center_canvas_y)
    M_rs = cv2.getRotationMatrix2D(center_tuple, angle_deg, scale)
    shear_x_rad = np.deg2rad(shear_x_deg)
    M_final = M_rs.copy()
    M_final[0,1] += np.tan(shear_x_rad) # This is an x-shear based on y from rotation center
    M_final[0,2] += trans_x
    M_final[1,2] += trans_y
    logger.debug(f"apply_affine - Final Matrix M_final:\n{M_final}")

    # Warp image
    warped_image = cv2.warpAffine(image, M_final, (w_canvas, h_canvas), # Output is oversized
                                  flags=cv2.INTER_LINEAR, # Smoother for image
                                  borderMode=cv2.BORDER_REFLECT_101)
                                  
    # Warp masks
    warped_masks_output = []
    if input_masks_list is not None:
        for i, mask_item in enumerate(input_masks_list):
            if mask_item is None: warped_masks_output.append(None); continue

            # --- Ensure mask is uint8 and single channel ---
            current_mask_for_warp = mask_item
            if current_mask_for_warp.ndim == 3 and current_mask_for_warp.shape[2] == 1:
                current_mask_for_warp = current_mask_for_warp[:, :, 0]
            elif current_mask_for_warp.ndim == 3:
                logger.warning(f"Mask item {i} has unexpected shape {current_mask_for_warp.shape}. Taking first channel.")
                current_mask_for_warp = current_mask_for_warp[:, :, 0]

            if current_mask_for_warp.dtype != np.uint8:
                logger.debug(f"Mask item {i} dtype is {current_mask_for_warp.dtype}. Converting to uint8 for warp.")
                # Assuming mask values are 0 or >0 for feature
                current_mask_for_warp = (current_mask_for_warp > 0).astype(np.uint8)
            # --- End Ensure ---

            logger.debug(f"Warping mask item {i} (shape: {current_mask_for_warp.shape}, dtype: {current_mask_for_warp.dtype})")

            # --- THE ACTUAL WARP ---
            warped_mask = cv2.warpAffine(current_mask_for_warp, M_final, (w_canvas, h_canvas),
                                         flags=cv2.INTER_NEAREST,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=0) # Explicit scalar 0
                                         
            save_debug_mask(warped_mask, "apply_affine", i, "WARPED_OUTPUT", current_call_count, logger) # DEBUG
            # --- DEBUG: Check sum immediately after warp ---
            logger.debug(f"  Mask item {i} sum AFTER warp (before appending): {warped_mask.sum()}")
            warped_masks_output.append(warped_mask)


    # Calculate warp field for the oversized canvas
    warp_field = None
    try:
        M_inv = cv2.invertAffineTransform(M_final)
        yy, xx = np.indices((h_canvas, w_canvas), dtype=np.float32) # Use oversized dimensions
        coords_flat = np.stack([xx.ravel(), yy.ravel(), np.ones(h_canvas*w_canvas)], axis=1)
        original_coords_flat = coords_flat @ M_inv.T
        original_x = original_coords_flat[:, 0].reshape(h_canvas, w_canvas)
        original_y = original_coords_flat[:, 1].reshape(h_canvas, w_canvas)
        warp_dy = yy - original_y
        warp_dx = xx - original_x
        warp_field = np.stack((warp_dy, warp_dx), axis=-1)
    except Exception as e_wf:
        logger.error(f"Could not calculate warp field for affine: {e_wf}", exc_info=True)

    return warped_image, warped_masks_output, warp_field


def apply_elastic(image, input_masks_list, params, rng, logger):
    """Applies elastic mesh deformation using skimage.transform.warp."""
    # DEBUG
    global DEBUG_CALL_COUNT
    call_key = f"elastic_{params.get('name_suffix', 'step')}"
    DEBUG_CALL_COUNT[call_key] = DEBUG_CALL_COUNT.get(call_key, 0) + 1
    current_call_count = DEBUG_CALL_COUNT[call_key]

    logger.info(f"--- apply_elastic Call #{current_call_count} ---")
    if input_masks_list:
        for i, m_in in enumerate(input_masks_list):
            save_debug_mask(m_in, "apply_elastic", i, "INPUT", current_call_count, logger)
    # ------
    logger.info(f"--- apply_elastic CALLED --- Params: {params}")
    if image is None:
        logger.warning("apply_elastic received None for image. Skipping.")
        return None, [None] * len(input_masks_list) if input_masks_list else [], None

    h_canvas, w_canvas = image.shape[:2] # Dimensions of the input (oversized) canvas
    alpha = params.get('alpha', 0)
    sigma = params.get('sigma', 1)
    grid_scale = params.get('grid_scale', 4)
    logger.info(f"apply_elastic - Canvas: {w_canvas}x{h_canvas}, Alpha:{alpha:.1f}, Sigma:{sigma:.1f}, GridScale:{grid_scale}")

    np_rng = np.random.RandomState(rng.randint(0, 2**32 - 1))
    dh = max(1, h_canvas // grid_scale)
    dw = max(1, w_canvas // grid_scale)
    # Displacement fields (dx, dy) for each pixel
    map_x_coarse = gaussian_filter( (np_rng.rand(dh, dw) * 2 - 1), sigma, mode="reflect", cval=0) * alpha
    map_y_coarse = gaussian_filter( (np_rng.rand(dh, dw) * 2 - 1), sigma, mode="reflect", cval=0) * alpha
    # Resize displacement fields to full canvas size
    map_x = cv2.resize(map_x_coarse, (w_canvas, h_canvas), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(map_y_coarse, (w_canvas, h_canvas), interpolation=cv2.INTER_LINEAR)

    logger.debug(f"apply_elastic - Displacement field map_x range: [{np.min(map_x):.2f}, {np.max(map_x):.2f}]")
    logger.debug(f"apply_elastic - Displacement field map_y range: [{np.min(map_y):.2f}, {np.max(map_y):.2f}]")

    # Create sampling indices for skimage.transform.warp
    # indices[0,i,j] = y_new, indices[1,i,j] = x_new
    y_coords, x_coords = np.meshgrid(np.arange(h_canvas), np.arange(w_canvas), indexing='ij')
    # For each output pixel (y_coords, x_coords), find where to sample from in input: (y_coords + map_y, x_coords + map_x)
    indices = np.stack([y_coords + map_y, x_coords + map_x], axis=0) # Shape (2, H, W)

    # Warp image
    warped_image = warp(image, indices, order=1, mode='reflect', cval=0, preserve_range=True,
                        output_shape=(h_canvas, w_canvas)).astype(image.dtype)

    # Warp masks
    warped_masks_output = []
    if input_masks_list is not None:
        for i, mask_item in enumerate(input_masks_list):
            if mask_item is None: warped_masks_output.append(None); continue
            
            current_mask_for_warp = mask_item
            if current_mask_for_warp.ndim == 3 and current_mask_for_warp.shape[2] == 1:
                current_mask_for_warp = current_mask_for_warp[:, :, 0]
            elif current_mask_for_warp.ndim == 3 :
                logger.warning(f"Elastic Mask item {i} has unexpected shape {current_mask_for_warp.shape}. Taking first channel.")
                current_mask_for_warp = current_mask_for_warp[:,:,0]


            # For skimage.transform.warp, input dtype for masks is often preserved,
            # but ensure cval=0 is appropriate for uint8 mask.
            if current_mask_for_warp.dtype != np.uint8:
                current_mask_for_warp = (current_mask_for_warp > 0).astype(np.uint8)


            warped_mask = warp(current_mask_for_warp, indices, order=0, mode='constant', cval=0,
                               preserve_range=True, output_shape=(h_canvas, w_canvas)).astype(current_mask_for_warp.dtype)
            logger.debug(f"  Elastic Mask item {i} sum AFTER warp (before appending): {warped_mask.sum()}")
            save_debug_mask(warped_mask, "apply_elastic", i, "WARPED_OUTPUT", current_call_count, logger)
            warped_masks_output.append(warped_mask)

    # The displacement field itself is (map_y, map_x)
    warp_field = np.stack((map_y, map_x), axis=-1) # Displacement (dy, dx)

    return warped_image, warped_masks_output, warp_field