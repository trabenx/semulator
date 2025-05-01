import numpy as np
import cv2
from skimage.draw import disk, rectangle

def apply_edge_ripple(mask, params, rng):
    """Applies ripple to shape edges using contour perturbation."""
    amplitude = params.get('amplitude', 1.5)
    # Adjust frequency interpretation: higher value means more ripples
    frequency_factor = params.get('frequency_factor', 10) # How many ripples per unit length approx
    noise_factor = params.get('noise_factor', 1.0) # Add randomness to ripple

    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # Need all points
    if not contours: return mask

    output_mask = np.zeros_like(mask)
    for contour in contours:
        if len(contour) < 3: continue # Need at least 3 points

        perturbed_contour_points = []
        contour_length = cv2.arcLength(contour.astype(np.float32), closed=True)
        if contour_length < 1e-6: continue # Skip degenerate contours

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
    count = params.get('count', 2)
    size_fraction = params.get('size_fraction', 0.1)
    hole_probability = params.get('hole_probability', 0.6)

    # Find connected components (individual shapes) first
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    output_mask = mask.copy()

    # Iterate through each component (shape), skipping background label 0
    for label_id in range(1, num_labels):
        component_mask = (labels_im == label_id)
        # Get bounding box from stats
        x, y, w, h, area = stats[label_id]
        if w < 3 or h < 3 or area < 5: continue # Skip tiny components

        for _ in range(count):
            # Find a random point *inside* this specific component
            attempts = 0
            max_attempts = 50 # Increase attempts
            while attempts < max_attempts:
                # Choose random point within the bounding box
                defect_cx = rng.randint(x, x + w)
                defect_cy = rng.randint(y, y + h)
                # Check if point is inside the component mask using the labels image
                if labels_im[defect_cy, defect_cx] == label_id:
                    break
                attempts += 1
            if attempts == max_attempts:
                # print(f"Warning: Could not find point inside component {label_id} for break/hole.")
                continue # Failed to find suitable point for this defect

            # Determine defect size relative to component size
            max_defect_dim = max(3, int(min(w, h) * size_fraction)) # Ensure min size 3
            defect_size = rng.randint(max(1, max_defect_dim // 3), max_defect_dim) # Range for defect size

            if rng.random() < hole_probability: # Draw hole (circle)
                radius = max(1, defect_size // 2)
                cv2.circle(output_mask, (defect_cx, defect_cy), radius, 0, -1) # Draw black circle
            else: # Draw break (rectangle)
                angle = rng.uniform(0, 180)
                rect_w = defect_size
                rect_h = max(1, rng.randint(max(1, defect_size//4), max(1, defect_size//2))) # Make it elongated
                # Ensure dimensions are positive
                rect_w = max(1, rect_w)
                rect_h = max(1, rect_h)
                box = cv2.boxPoints(((defect_cx, defect_cy), (rect_w, rect_h), angle))
                cv2.drawContours(output_mask, [box.astype(int)], 0, 0, -1) # Draw filled black rectangle

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
    if patch.size == 0: return mask

    # Apply elastic to the patch (masks_to_warp is just the patch itself)
    warped_patch, _, _ = apply_elastic(patch, [patch], params, rng)

    # Create output mask and place warped patch back
    output_mask = mask.copy()
    output_mask[y0:y1, x0:x1] = warped_patch[0] # warped_masks_out returns a list

    # Optional: Ensure result doesn't exceed original area significantly?
    # output_mask &= mask # Intersect? Might remove desired ripple effects.
    # Maybe dilate original mask slightly for masking?
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin//2, margin//2))
    dilated_original = cv2.dilate(mask, kernel)
    output_mask &= dilated_original # Keep warp within dilated original area


    return output_mask


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


# --- Factory (if needed, or call directly in generator) ---
# Factory function might be less useful here as inputs differ (mask vs image_layer)
