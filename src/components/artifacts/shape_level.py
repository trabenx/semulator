import numpy as np
import cv2
import logging
from skimage.draw import disk, rectangle
from scipy.ndimage import map_coordinates, binary_erosion

logger = logging.getLogger(__name__)

def apply_edge_ripple(mask, params, rng):
    """Applies ripple to shape edges using contour perturbation."""
    amplitude = params.get('amplitude', 2.0) # Increased default example
    frequency_factor = params.get('frequency_factor', 12)
    noise_factor = params.get('noise_factor', 1.2) # Increased default example

    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # Need all points
    if not contours:
        return mask

    output_mask = np.zeros_like(mask)
    for contour in contours:
        if len(contour) < 3:
            continue # Need at least 3 points

        perturbed_contour_points = []
        contour_length = cv2.arcLength(contour.astype(np.float32), closed=True)
        if contour_length < 1e-3:
            continue # Skip degenerate contours

        # Calculate approx distance along contour for frequency calculation
        distances = np.zeros(len(contour))
        for i in range(1, len(contour)):
             distances[i] = distances[i-1] + np.linalg.norm(contour[i][0] - contour[i-1][0])

        for i in range(len(contour)):
            p_prev = contour[i-1][0]
            p_curr = contour[i][0]
            p_next = contour[(i+1) % len(contour)][0]

            # Tangent approx: vector from prev to next
            tangent = p_next.astype(float) - p_prev.astype(float)
            # Normal: rotate tangent 90 deg and normalize
            normal = np.array([-tangent[1], tangent[0]])
            norm_mag = np.linalg.norm(normal)
            if norm_mag > 1e-6:
                 normal = normal / norm_mag
            else: # Handle coincident points by looking further? Simple fallback for now.
                normal = np.array([0.0, 0.0]) # Or use fallback logic
                # More robust: check p_curr vs p_prev or p_curr vs p_next

            # Calculate displacement
            # Use distance along contour for sinusoidal part
            angle = (distances[i] / contour_length) * frequency_factor * 2 * np.pi
            base_displacement = amplitude * np.sin(angle)
            # Add noise component
            noise_displacement = rng.uniform(-amplitude * 0.5, amplitude * 0.5) * noise_factor
            total_displacement = base_displacement + noise_displacement

            perturbed_point = p_curr + normal * total_displacement
            perturbed_contour_points.append([perturbed_point.astype(int)]) # Append list for drawContours format

        if perturbed_contour_points:
            cv2.drawContours(output_mask, [np.array(perturbed_contour_points)], -1, 1, -1) # Draw filled

    return output_mask


def apply_breaks_holes(mask, params, rng):
    """Introduces random holes (circular) or breaks (rectangular) into shapes."""
    count = params.get('count', 3)
    size_fraction = params.get('size_fraction', 0.1)
    hole_probability = params.get('hole_probability', 0.6)
    min_defect_size = 5 # Minimum pixels for width/height/radius
    
    # Get mask dimensions needed for boundary checks
    mask_h, mask_w = mask.shape[:2]
    
    # Find connected components (individual shapes) first
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1: return mask # No foreground components found

    output_mask = mask.copy()

    # Iterate through each component (shape), skipping background label 0
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        if w < min_defect_size or h < min_defect_size or area < (min_defect_size*min_defect_size):
            continue

        for _ in range(count):
            # Find a random point *inside* this specific component
            attempts = 0
            max_attempts = 50
            found_point = False
            while attempts < max_attempts:
                # Choose random point within the bounding box
                defect_cx = rng.randrange(x, x + w) # Generates x <= cx < x + w
                defect_cy = rng.randrange(y, y + h) # Generates y <= cy < y + h

                # Boundary check (should be redundant now but safe)
                if 0 <= defect_cy < mask_h and 0 <= defect_cx < mask_w:
                    # Check if point is inside the *specific component* mask
                    if labels_im[defect_cy, defect_cx] == label_id:
                        found_point = True
                        break # Found a valid point
                attempts += 1

            if not found_point:
                # logger.debug(f"Could not find point inside component {label_id} for break/hole after {max_attempts} attempts.")
                continue # Skip creating defect if no point found

            # Determine defect size
            max_defect_dim = max(min_defect_size, int(min(w, h) * size_fraction))
            # Ensure upper bound of randint is >= lower bound
            defect_size = rng.randint(min_defect_size, max(min_defect_size, max_defect_dim)) # Use randint correctly

            if rng.random() < hole_probability: # Draw hole (circle)
                radius = max(1, defect_size // 2)
                cv2.circle(output_mask, (defect_cx, defect_cy), radius, 0, -1)
            else: # Draw break (rectangle)
                angle = rng.uniform(0, 180)
                rect_w = max(1, defect_size)
                # Make rectangle slightly elongated
                rect_h = max(1, rng.randint(max(1, defect_size//3), max(1, defect_size)))
                try:
                    box = cv2.boxPoints(((defect_cx, defect_cy), (rect_w, rect_h), angle))
                    cv2.drawContours(output_mask, [box.astype(int)], 0, 0, -1)
                except Exception as e: # Catch potential errors in boxPoints/drawContours
                     logger.warning(f"Error drawing break rectangle: {e}")

    return output_mask


def apply_local_elastic(mask, params, rng):
    """Applies elastic deformation localized to the mask area."""
    # This is an approximation. True local elastic requires warping only inside.
    # Approach: Apply global elastic, but only keep changes within original mask bounds.
    # Or generate warp field only near mask? Harder.
    # Simpler: Apply global elastic, then mask the result with original mask area (+ a bit).

    # Use the global elastic function but maybe with smaller params
    from .geometric import apply_elastic # Avoid circular import if possible

    # Get bounding box + margin
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return mask
    x, y, w, h = cv2.boundingRect(np.concatenate(contours))
    margin = max(w,h)//4 # Margin around bounding box
    x0 = max(0, x-margin)
    y0 = max(0, y-margin)
    x1 = min(mask.shape[1], x+w+margin)
    y1 = min(mask.shape[0], y+h+margin)

    # Extract patch
    patch = mask[y0:y1, x0:x1]
    if patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0: return mask

    # Apply elastic to the patch (masks_to_warp is just the patch itself)
    try:
        warped_patch_list, _, _ = apply_elastic(patch, [patch], params, rng) # Call global elastic
        if not warped_patch_list:
            return mask # Handle case where warp failed
        warped_patch = warped_patch_list[0]
        output_mask = mask.copy()
        # Ensure dimensions match before placing back
        h_patch, w_patch = warped_patch.shape[:2]
        output_mask[y0:y0+h_patch, x0:x0+w_patch] = warped_patch
        # Masking logic (optional, as before)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin//2, margin//2))
        # dilated_original = cv2.dilate(mask, kernel)
        # output_mask &= dilated_original
        return output_mask
    except Exception as e:
        logger.error(f"Error during apply_local_elastic warp: {e}", exc_info=True)
        return mask # Return original mask on error


def apply_contour_smoothing(mask, params, rng):
    """Smooths shape contours using Gaussian blur on the mask."""
    ksize = params.get('kernel_size', 5)
    if ksize % 2 == 0: ksize += 1 # Ensure odd kernel size

    # Blur the mask
    blurred_mask = cv2.GaussianBlur(mask.astype(float), (ksize, ksize), 0)

    # Threshold back to binary (adjust threshold if needed, 0.5 is common)
    smoothed_mask = (blurred_mask > 0.5).astype(np.uint8)

    return smoothed_mask


def apply_local_brightness(image_layer, mask, params, rng):
     """Applies brightness variation based on Perlin noise, simulating thickness."""
     # Modifies the float image layer *before* composition
     contrast = params.get('contrast', 0.1)
     scale = params.get('scale', 20)

     try:
          from perlin_noise import PerlinNoise
          HAS_PERLIN = True
     except ImportError:
          HAS_PERLIN = False

     if not HAS_PERLIN or mask is None or image_layer is None:
          return image_layer # No change if Perlin unavailable or no input

     h, w = image_layer.shape
     noise_gen = PerlinNoise(octaves=4, seed=rng.randint(0, 10000))

     # Generate noise map matching image size
     brightness_noise = np.zeros((h,w), dtype=np.float32)
     for i in range(h):
          for j in range(w):
               brightness_noise[i,j] = noise_gen([i/scale, j/scale])

     # Normalize noise -1 to 1, scale by contrast
     brightness_noise = (brightness_noise - np.mean(brightness_noise))
     std_dev = np.std(brightness_noise)
     if std_dev > 1e-6:
          brightness_noise = brightness_noise / std_dev
     brightness_variation = brightness_noise * contrast

     # Apply multiplicatively only within the mask area
     output_layer = image_layer.copy()
     # Apply variation around 1.0 multiplicatively
     output_layer[mask > 0] *= (1.0 + brightness_variation[mask > 0])
     output_layer = np.clip(output_layer, 0.0, 1.0) # Ensure valid range

     return output_layer


def apply_etch_bias(mask, params, rng):
    """Applies uniform erosion or dilation to the mask."""
    # Amount is in pixels: negative=erode, positive=dilate
    amount = params.get('amount', rng.uniform(-2.0, 2.0)) # Use range from params if present
    amount_int = int(round(amount))

    if amount_int == 0:
        return mask # No change

    # Kernel size must be odd and positive
    k_size = abs(amount_int) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    if amount_int < 0: # Erosion
        biased_mask = cv2.erode(mask, kernel, iterations=1)
        logger.debug(f"Applied etch bias (erosion): amount={amount_int}")
    else: # Dilation
        biased_mask = cv2.dilate(mask, kernel, iterations=1)
        logger.debug(f"Applied etch bias (dilation): amount={amount_int}")

    return biased_mask


def apply_local_affine(mask, params, rng):
    """
    Applies a small, randomized affine transformation centered on the shape mask.
    Much faster than elastic deformation for per-instance variation.
    """
    if np.sum(mask) < 10: # Skip tiny masks where transform is pointless/problematic
        return mask

    # Get max variation parameters from config
    max_scale_delta = params.get('max_scale_delta', 0.03)
    max_rot_deg = params.get('max_rotation_deg', 3)
    max_shear_deg = params.get('max_shear_deg', 3)
    max_trans_frac = params.get('max_translate_fraction', 0.03)

    # --- Calculate transformation center (centroid or bounding box center) ---
    # Using boundingRect center is usually safer/faster than moments for binary masks
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return mask # Should not happen if sum > 0, but safety check
    x, y, w, h = cv2.boundingRect(contours[0]) # Use first contour's bbox
    center_x = x + w / 2
    center_y = y + h / 2
    center = (center_x, center_y)

    # --- Generate random parameters within limits ---
    scale = 1.0 + rng.uniform(-max_scale_delta, max_scale_delta)
    angle = rng.uniform(-max_rot_deg, max_rot_deg)
    shear = rng.uniform(-max_shear_deg, max_shear_deg)
    # Translate relative to shape size
    trans_x = rng.uniform(-max_trans_frac, max_trans_frac) * w
    trans_y = rng.uniform(-max_trans_frac, max_trans_frac) * h

    # --- Get Affine Matrix ---
    M_rot_scale = cv2.getRotationMatrix2D(center, angle, scale)
    # Add translation
    M_rot_scale[0, 2] += trans_x
    M_rot_scale[1, 2] += trans_y
    # Add shear (approximate, applied relative to center after rot/scale/trans)
    # This requires modifying the matrix elements carefully.
    shear_rad = np.deg2rad(shear)
    shear_matrix = np.array([[1, np.tan(shear_rad), 0],
                             [0, 1,              0]])
    # To apply shear relative to center, we need to translate center to origin,
    # shear, then translate back. Or modify M_rot_scale directly (more complex).
    # Let's modify M_rot_scale - check matrix math:
    # M = T(center) * Shear * T(-center) * M_rot_scale_trans (approx order)
    # Simpler approximation for small shear: Add shear term directly?
    # M_rot_scale[0, 1] += np.tan(shear_rad) # Simple addition, might not be perfectly centered shear

    # --- Alternative: Build matrix step-by-step (more robust centering) ---
    # 1. Translate center to origin
    T1 = np.array([[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]], dtype=float)
    # 2. Scale
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=float)
    # 3. Shear
    Sh = np.array([[1, np.tan(shear_rad), 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    # 4. Rotate
    angle_rad_rot = np.deg2rad(angle)
    cos_a, sin_a = np.cos(angle_rad_rot), np.sin(angle_rad_rot)
    R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=float)
    # 5. Translate back to center AND apply random translation
    T2 = np.array([[1, 0, center_x + trans_x], [0, 1, center_y + trans_y], [0, 0, 1]], dtype=float)

    # Combine matrices: M = T2 * R * Sh * S * T1
    M_combined = T2 @ R @ Sh @ S @ T1
    # Get the final 2x3 matrix for warpAffine
    M_final = M_combined[0:2, 0:3]
    # --- End Alternative Matrix Building ---


    # --- Apply Transformation ---
    # Use INTER_NEAREST for masks to avoid creating intermediate gray values
    # Use BORDER_CONSTANT with value 0 (black background)
    h_img, w_img = mask.shape
    warped_mask = cv2.warpAffine(mask, M_final, (w_img, h_img),
                                 flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)

    # logger.debug(f"Applied local affine: scale={scale:.3f}, angle={angle:.1f}, shear={shear:.1f}, trans=({trans_x:.1f},{trans_y:.1f})") # Optional debug
    return warped_mask


def apply_shape_border(rendered_instance, mask, params, target_intensity, rng):
    """
    Modifies the intensity of the border region of a rendered shape instance.

    Args:
        rendered_instance (np.ndarray): Float array (HxW) of the already rendered shape.
        mask (np.ndarray): uint8 binary mask corresponding to the shape.
        params (dict): Dictionary with 'thickness' and 'intensity_factor'.
        target_intensity (float): The original target intensity of the shape (before alpha).
        rng (random.Random): Random number generator.

    Returns:
        np.ndarray: The modified rendered_instance array.
    """
    thickness = params.get('thickness', 1) # Border thickness in pixels
    # Intensity factor: >1 = brighter border, <1 = darker border, relative to target_intensity
    intensity_factor = params.get('intensity_factor', 1.5)

    thickness = max(1, int(round(thickness))) # Ensure positive integer thickness

    if np.sum(mask) < 10: # Skip if mask is too small
        return rendered_instance

    # --- Find the border region ---
    # Erode the mask. The border is where the original mask is 1 but the eroded mask is 0.
    # Structure determines connectivity (4 or 8) for erosion
    structure = np.array([[0,1,0], [1,1,1], [0,1,0]], dtype=bool) # 4-connectivity (more controlled)
    # structure = np.ones((3,3), dtype=bool) # 8-connectivity
    try:
        # Erode requires integer iterations based on thickness
        inner_mask = binary_erosion(mask, structure=structure, iterations=thickness, border_value=0)
        border_mask = (mask > 0) & (~inner_mask) # XOR is not quite right, use AND NOT
    except Exception as e:
         logger.error(f"Error during border erosion (thickness={thickness}): {e}", exc_info=True)
         return rendered_instance # Return unchanged on error

    if np.sum(border_mask) == 0: # No border found (maybe shape too thin)
        return rendered_instance

    # --- Calculate border intensity ---
    # Apply factor to the original target intensity *before* alpha might have reduced it
    border_intensity = target_intensity * intensity_factor
    border_intensity = np.clip(border_intensity, 0.0, 1.0) # Clamp to valid range

    # --- Modify the rendered instance ---
    output_render = rendered_instance.copy()
    output_render[border_mask] = border_intensity # Set border pixels to new intensity

    # logger.debug(f"Applied shape border: thickness={thickness}, factor={intensity_factor:.2f}, border_intensity={border_intensity:.2f}") # Debug
    return output_render

# --- Factory (if needed, or call directly in generator) ---
# Factory function might be less useful here as inputs differ (mask vs image_layer)
