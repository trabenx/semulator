import numpy as np
import cv2
from src.core.utils import get_kernel
from scipy.ndimage import convolve1d, binary_erosion, gaussian_filter
import logging


def apply_psf_blur(image, params, rng, logger):
    """Applies simulated probe PSF blur (Gaussian/elliptical)."""
    sigma = params.get('sigma', 1.0)
    astigmatism_ratio = params.get('astigmatism_ratio', 1.0)
    angle = params.get('angle', 0) # Degrees

    sigma_x = sigma
    # Ensure ratio >= 1 for convention sigma_y >= sigma_x
    astigmatism_ratio = max(1.0, astigmatism_ratio)
    sigma_y = sigma * astigmatism_ratio
    max_sigma = sigma_y # Since ratio >= 1

    # Determine kernel size (ensure it's large enough for the max sigma)
    ksize = int(max_sigma * 6 + 1)
    if ksize % 2 == 0: ksize += 1
    ksize = max(3, ksize) # Ensure minimum size of 3

    # Create potentially rotated elliptical kernel using the utility function
    kernel = get_kernel(ksize, sigma_x, sigma_y, angle)

    # Apply convolution using filter2D
    # Use BORDER_REFLECT_101 (reflect without repeating border pixel) or BORDER_REPLICATE
    blurred = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)

    logger.debug(f"Applied PSF blur: sigma={sigma:.2f}, ratio={astigmatism_ratio:.2f}, angle={angle:.1f}")
    return blurred # filter2D preserves float type


def apply_defocus_blur(image, params, rng, logger):
    """Applies spatially varying defocus blur."""
    max_radius = params.get('max_radius', 2.0) # Max sigma of blur
    angle = params.get('gradient_angle_deg', rng.uniform(0, 360)) # Direction of focus gradient

    h, w = image.shape
    output_image = np.zeros_like(image)

    # Create a map of blur sigmas based on gradient direction
    rad_angle = np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad_angle), np.sin(rad_angle)
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    xx, yy = np.meshgrid(x, y)
    # Project coordinates onto gradient direction, map range to [0, 1]
    projection = (xx * cos_a + yy * sin_a)
    focus_map = (projection - np.min(projection)) / (np.max(projection) - np.min(projection) + 1e-6) # Range 0 to 1

    # Sigma map from focus map (0=sharp, 1=max_radius)
    sigma_map = focus_map * max_radius

    # --- Apply blur ---
    # Method 1: Quantize sigma levels and interpolate (simpler)
    num_levels = 5 # Number of discrete blur levels
    quantized_sigmas = np.linspace(1e-3, max_radius, num_levels) # Avoid sigma=0
    blurred_layers = []
    for sigma in quantized_sigmas:
         ksize = int(sigma * 6 + 1)
         if ksize % 2 == 0: ksize += 1
         blurred_layers.append(cv2.GaussianBlur(image, (ksize, ksize), sigma))

    # Interpolate based on sigma_map
    # Find which two layers each pixel falls between
    sigma_map_scaled = sigma_map * (num_levels - 1) # Map sigma to index range [0, num_levels-1]
    idx0 = np.floor(sigma_map_scaled).astype(int)
    idx1 = np.ceil(sigma_map_scaled).astype(int)
    # Ensure indices are within bounds
    idx0 = np.clip(idx0, 0, num_levels - 1)
    idx1 = np.clip(idx1, 0, num_levels - 1)

    # Calculate interpolation weights (fractional part of index)
    weight1 = sigma_map_scaled - idx0
    weight0 = 1.0 - weight1

    # Perform interpolation
    output_image = blurred_layers[0] * 0 # Initialize (needed if using index arrays)
    for i in range(num_levels):
         mask0 = (idx0 == i)
         output_image[mask0] += blurred_layers[i][mask0] * weight0[mask0]
         mask1 = (idx1 == i)
         output_image[mask1] += blurred_layers[i][mask1] * weight1[mask1]


    # Method 2: Per-pixel filtering (Very slow) - Not implemented

    logger.debug(f"Applied spatially varying defocus: max_radius={max_radius:.2f}, angle={angle:.1f}")
    return np.clip(output_image, 0.0, 1.0)


def apply_charging(image, params, rng, logger):
    """Simulates charging artifacts (brightening/streaking near bright edges)."""
    intensity_factor = params.get('intensity', 0.15) # How much to brighten/streak
    edge_threshold_range = params.get('edge_threshold_range', [0.6, 0.9]) # Range for brightness threshold at edges
    edge_gradient_threshold = params.get('edge_gradient_threshold', 0.1) # Min gradient magnitude
    radius = max(1, int(params.get('radius', 3))) # Radius for charge accumulation blur/dilation
    length_factor = params.get('length_factor', 15) # How far streaks extend relative to radius
    streak_direction = rng.uniform(-15, 15) # Angle deviation from horizontal (degrees) for streaks

    h, w = image.shape
    output_image = image.copy()

    # --- 1. Detect potential charging sites (bright edges) ---
    # Calculate gradients
    sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Find pixels that are bright AND have significant gradient
    brightness_threshold = rng.uniform(edge_threshold_range[0], edge_threshold_range[1])
    charge_sites_mask = (image > brightness_threshold) & (gradient_magnitude > edge_gradient_threshold)

    # Optional: Erode slightly to pinpoint the brightest part of the edge
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    charge_sites_mask = binary_erosion(charge_sites_mask, structure=kernel_erode)
    charge_sites_mask = charge_sites_mask.astype(np.float32) # Convert to float for filtering
    
    # Optional: Erode slightly (Now binary_erosion is available)
    if np.sum(charge_sites_mask) > 0: # Only erode if there are sites
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        # Ensure input to binary_erosion is boolean
        charge_sites_mask = binary_erosion(charge_sites_mask, structure=kernel_erode)
        charge_sites_mask = charge_sites_mask.astype(np.float32) # Convert back for filtering
    else: # No sites detected, convert empty mask to float
        charge_sites_mask = charge_sites_mask.astype(np.float32)

    if np.sum(charge_sites_mask) < 1: # No charging sites detected
        return output_image

    # --- 2. Simulate charge accumulation bloom/glow ---
    # Blur the charge sites mask to create a glow effect
    glow_ksize = radius * 2 + 1
    charge_glow = cv2.GaussianBlur(charge_sites_mask, (glow_ksize, glow_ksize), radius)
    # Scale glow intensity
    if np.max(charge_glow) > 1e-6: # Avoid division by zero if glow is all zeros
        charge_glow = charge_glow / np.max(charge_glow)
    # Add glow to the image
    output_image += charge_glow * intensity_factor * 0.7 # Glow is additive

    # --- 3. Simulate streaking ---
    streak_length = max(3, int(radius * length_factor))
    if streak_length > 1:
        # Create a directional motion blur kernel based on streak_direction
        angle_rad = np.deg2rad(streak_direction)
        dx, dy = np.cos(angle_rad), np.sin(angle_rad)

        # Create kernel line points
        center_x, center_y = streak_length // 2, streak_length // 2
        kernel_motion = np.zeros((streak_length, streak_length), dtype=np.float32)
        points = []
        for i in range(streak_length):
             x = int(center_x + (i - streak_length / 2) * dx)
             y = int(center_y + (i - streak_length / 2) * dy)
             # Clip points to kernel bounds
             if 0 <= x < streak_length and 0 <= y < streak_length:
                  points.append((x,y))

        if points:
             # Set kernel values along the line (simple averaging)
             line_points = np.array(points)
             kernel_motion[line_points[:, 1], line_points[:, 0]] = 1.0
             kernel_motion /= np.sum(kernel_motion) # Normalize

             if np.sum(kernel_motion) > 1e-6:
                 kernel_motion /= np.sum(kernel_motion) # Normalize
             else:
                 kernel_motion = None # Avoid using zero kernel


             if kernel_motion is not None:
                 # Apply directional blur to the charge *sites* mask (not the glow)
                 streaks = cv2.filter2D(charge_sites_mask, -1, kernel_motion)
                 if np.max(streaks) > 1e-6:
                     streaks = streaks / np.max(streaks) # Normalize approx 0-1

             # Add streaks to the image
             output_image += streaks * intensity_factor * 1.0 # Streaks can be stronger

    logger.debug(f"Applied charging simulation: intensity={intensity_factor:.2f}, threshold={brightness_threshold:.2f}")
    return np.clip(output_image, 0.0, 1.0)


def apply_topographic_shading(image, layers_data, params, rng, logger):
    """Applies shading based on synthetic height map."""
    strength = params.get('strength', 0.3)
    light_angle_deg = params.get('light_angle_deg', 45.0)
    height_scale = params.get('height_scale', 50.0) # For Perlin noise if used
    use_layer_height_prob = params.get('use_layer_height', 0.3)

    h, w = image.shape
    height_map = np.zeros((h, w), dtype=np.float32)

    # --- Generate Height Map ---
    # Option 1: Based on layer structure (simple stacking)
    if rng.random() < use_layer_height_prob and layers_data:
        logger.debug("Generating height map from layers...")
        # Assign height based on layer index (higher index = higher height)
        layer_height_step = 1.0 / (len(layers_data) + 1)
        # Sort by layer index to ensure correct stacking order
        sorted_indices = sorted(layers_data.keys())
        current_height = 0.0
        for idx in sorted_indices:
            mask = layers_data[idx]['actual_mask'] # Use actual mask post-shape artifacts
            current_height += layer_height_step
            height_map[mask > 0] = current_height
        # Normalize height map 0-1
        if np.max(height_map) > 1e-6:
             height_map /= np.max(height_map)

    # Option 2: Random Perlin noise height map
    else:
        logger.debug("Generating height map from Perlin noise...")
        try:
             from perlin_noise import PerlinNoise
             HAS_PERLIN = True
        except ImportError:
             HAS_PERLIN = False
             logger.warning("Warning: Perlin noise library not found for topographic shading.")

        if HAS_PERLIN:
             noise_gen = PerlinNoise(octaves=6, seed=rng.randint(0, 10000))
             for i in range(h):
                 for j in range(w):
                     height_map[i, j] = noise_gen([i / height_scale, j / height_scale])
             # Normalize noise to 0-1 range
             min_h, max_h = np.min(height_map), np.max(height_map)
             if max_h - min_h > 1e-6:
                 height_map = (height_map - min_h) / (max_h - min_h)
        else: # No height map possible
             return image


    # --- Calculate Shading ---
    # Use gradient of height map to simulate surface normals
    dx = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3)

    # Normal vector approx: (-dx, -dy, 1) - normalize Z component implicitly
    # Light vector based on angle (in xy plane, pointing towards light)
    light_angle_rad = np.deg2rad(light_angle_deg)
    light_x = np.cos(light_angle_rad)
    light_y = np.sin(light_angle_rad)
    light_z = 1.0 # Assume light source elevation for simplicity

    # Dot product: normal . light = (-dx * light_x) + (-dy * light_y) + (1 * light_z)
    # Simplified Lambertian: intensity proportional to dot product (clamped > 0)
    shading = (-dx * light_x - dy * light_y + light_z)
    # Normalize dot product (approx) and scale by strength
    # Basic scaling: map range to [1-strength, 1+strength] ?
    # Or simpler: add scaled shading centered around 0
    shading_norm = (shading - np.mean(shading)) # Center around 0
    std_dev = np.std(shading_norm)
    if std_dev > 1e-6:
         shading_norm = shading_norm / std_dev # Normalize std dev

    shading_effect = shading_norm * strength

    # Apply shading additively (or multiplicatively?)
    output_image = np.clip(image + shading_effect, 0.0, 1.0)

    logger.debug(f"Applied topographic shading: strength={strength:.2f}, light_angle={light_angle_deg:.1f}")
    return output_image, height_map


def apply_gradient_illumination(image, params, rng, logger):
    """Applies a gradual brightness gradient across the image."""
    max_delta = params.get('max_delta', 0.2) # Max brightness change (0 to max_delta)
    angle = params.get('angle', rng.uniform(0, 360)) # Gradient direction

    h, w = image.shape
    rad_angle = np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad_angle), np.sin(rad_angle)

    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    xx, yy = np.meshgrid(x, y)

    # Project coordinates, normalize range 0 to 1
    projection = (xx * cos_a + yy * sin_a)
    gradient_map = (projection - np.min(projection)) / (np.max(projection) - np.min(projection) + 1e-6)

    # Apply gradient additively, scaled by max_delta
    illumination_effect = (gradient_map - 0.5) * max_delta # Center effect around 0 change
    output_image = image + illumination_effect

    logger.debug(f"Applied gradient illumination: max_delta={max_delta:.2f}, angle={angle:.1f}")
    return np.clip(output_image, 0.0, 1.0)


def apply_striping_smearing(image, params, rng, logger):
     """Simulates simple striping or smearing via directional blur."""
     strength = params.get('strength', 0.03) # Corresponds approx to blur kernel size/effect
     direction = params.get('direction', rng.choice(['h', 'v']))

     ksize = max(3, int(strength * 100)) # Kernel size based on strength
     if ksize % 2 == 0: ksize +=1

     if direction == 'h': # Horizontal blur
          kernel = np.zeros((1, ksize), dtype=np.float32)
          kernel[0, :] = 1.0 / ksize
     else: # Vertical blur
          kernel = np.zeros((ksize, 1), dtype=np.float32)
          kernel[:, 0] = 1.0 / ksize

     blurred = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)

     # Mix blurred with original (or just return blurred?) Return blurred for stronger effect.
     # output_image = image * (1 - strength) + blurred * strength
     output_image = blurred

     logger.debug(f"Applied simple {direction}-smearing: strength={strength:.3f}")
     return output_image # Already clipped by blur


def apply_fixed_pattern_noise(image, params, rng, logger):
    """Adds low-frequency noise pattern."""
    scale = params.get('scale', 100.0)
    strength = params.get('strength', 0.02)

    try:
         from perlin_noise import PerlinNoise
         HAS_PERLIN = True
    except ImportError:
         HAS_PERLIN = False

    if not HAS_PERLIN:
         logger.warning("Warning: Perlin noise library not found for FPN.")
         return image, np.zeros_like(image) # Return zero noise map if skipped

    h, w = image.shape
    noise_gen = PerlinNoise(octaves=3, seed=rng.randint(0, 10000)) # Fewer octaves for smoother noise

    fpn_map = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
         for j in range(w):
             fpn_map[i, j] = noise_gen([i / scale, j / scale])

    # Normalize noise -1 to 1, scale by strength
    fpn_map = (fpn_map - np.mean(fpn_map))
    std_dev = np.std(fpn_map)
    if std_dev > 1e-6:
         fpn_map = fpn_map / std_dev
    added_noise = fpn_map * strength

    output_image = image + added_noise

    logger.debug(f"Added Fixed Pattern Noise: strength={strength:.3f}, scale={scale:.1f}")
    return output_image, added_noise


def apply_edge_brightness(image, params, rng, logger):
    """
    Adds brightness along the edges of features in the image.
    Simulates higher secondary electron yield at edges/slopes.
    """
    strength = params.get('strength', 0.3) # How much brighter edges become
    thickness = params.get('thickness', 1.5) # How thick the bright edge effect is (sigma for blur)
    # Canny edge detection thresholds
    low_thresh_factor = params.get('low_thresh_factor', 0.1) # Relative to max image intensity
    high_thresh_factor = params.get('high_thresh_factor', 0.3) # Relative to max image intensity

    # --- Detect Edges using Canny ---
    # Need image in 0-255 range for Canny
    if np.max(image) <= 1.0 and np.min(image) >= 0.0: # Check if likely float 0-1
        img_uint8 = (image * 255.0).astype(np.uint8)
    else: # Assume already scaled or convert differently if needed
         img_uint8 = np.clip(image, 0, 255).astype(np.uint8) # Clip and convert just in case


    # Calculate thresholds based on image intensity range (or use fixed if preferred)
    # Using simple percentile might be more robust than max
    # med_val = np.median(img_uint8)
    # low_thresh = int(max(0, (1.0 - 0.33) * med_val))
    # high_thresh = int(min(255, (1.0 + 0.33) * med_val))
    # Simpler: use factors of max intensity found
    max_val = np.max(img_uint8) if np.max(img_uint8) > 0 else 255
    low_thresh = int(max_val * low_thresh_factor)
    high_thresh = int(max_val * high_thresh_factor)

    edges = cv2.Canny(img_uint8, low_thresh, high_thresh, L2gradient=True)
    edges_float = edges.astype(np.float32) / 255.0 # Normalize edge map 0-1

    # --- Blur edges to create the "glow" effect ---
    # Sigma controls the thickness/spread of the brightness
    if thickness > 0.1:
         # Use SciPy Gaussian filter for float images
         blurred_edges = gaussian_filter(edges_float, sigma=thickness)
    else:
         blurred_edges = edges_float # No blur if thickness is negligible

    # Normalize blurred edges again (blurring might change max value)
    max_be = np.max(blurred_edges)
    if max_be > 1e-6:
        blurred_edges /= max_be

    # --- Add edge brightness to the original image ---
    # Additive effect, scaled by strength
    output_image = image + blurred_edges * strength

    # print(f"Applied edge brightness: strength={strength:.2f}, thickness={thickness:.2f}") # Debug
    return np.clip(output_image, 0.0, 1.0)
