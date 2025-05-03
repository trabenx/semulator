import numpy as np
import cv2
from ..core.utils import normalize_image
# Consider adding 'perlin-noise' to requirements.txt or implement manually
try:
    from perlin_noise import PerlinNoise
    HAS_PERLIN = True
except ImportError:
    HAS_PERLIN = False
    print("WARNING: 'perlin-noise' library not found. Perlin background disabled.")
import logging

logger = logging.getLogger(__name__)


def generate_background(config, size, magnification, rng):
    """Generates the background canvas."""
    bg_type = config.get('types')  # Get the type chosen during config randomization
    if not bg_type or not isinstance(bg_type, str): # Check if it's a valid string
        # Fallback if selected_type wasn't set (shouldn't happen with proper randomization)
        logger.error(f"Invalid or missing background type in randomized config: '{bg_type}'. Defaulting to flat.")
        bg_type = 'flat'

    params = config.get('parameters', {}).get(bg_type, {})
    height, width = size
    logger.debug(f"Generating background type: {bg_type} with params: {params}")

    background = np.zeros((height, width), dtype=np.float32) # Initialize


    if bg_type == 'flat':
        intensity = params.get('intensity', 0.2)
        background.fill(intensity)

    elif bg_type == 'gradient':
        start_intensity = params.get('start_intensity', 0.1)
        end_intensity = params.get('end_intensity', 0.3)
        angle = params.get('angle', 0)
        grad_type = params.get('type', 'linear')

        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)

        if grad_type == 'linear':
            rad_angle = np.deg2rad(angle)
            cos_a, sin_a = np.cos(rad_angle), np.sin(rad_angle)
            projection = xx * cos_a + yy * sin_a
            min_p, max_p = np.min(projection), np.max(projection)
            if max_p - min_p < 1e-6: gradient = np.zeros_like(projection)
            else: gradient = (projection - min_p) / (max_p - min_p)
            background = start_intensity + (end_intensity - start_intensity) * gradient
        elif grad_type == 'radial':
            center_x, center_y = 0.5, 0.5 # Assume center
            dist = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
            max_dist = np.sqrt(0.5**2 + 0.5**2)
            gradient = dist / max_dist
            background = start_intensity + (end_intensity - start_intensity) * gradient
        else:
            logger.warning(f"Unknown gradient type: {grad_type}. Using flat.")
            background.fill(start_intensity) # Use start_intensity for flat fallback

    elif bg_type == 'perlin':
         if not HAS_PERLIN:
             logger.warning("Perlin noise library not found. Using flat background.")
             intensity = params.get('intensity', 0.2) # Fallback intensity
             background.fill(intensity)
         else:
            octaves = params.get('octaves', 6)
            # Persistence/Lacunarity not directly used by perlin-noise lib
            # Scale noise freq relative to magnification
            scale = params.get('scale', 100.0) / max(0.1, magnification) # Avoid div by zero
            contrast = params.get('contrast', 0.1)
            base_intensity = params.get('base_intensity', 0.2)

            # Use the passed sample RNG for seed consistency
            noise_gen = PerlinNoise(octaves=octaves, seed=rng.randint(0, 2**32 - 1))

            noise_map = np.zeros((height, width), dtype=np.float32)
            for i in range(height):
                for j in range(width):
                    noise_map[i][j] = noise_gen([i / scale, j / scale])

            min_n, max_n = np.min(noise_map), np.max(noise_map)
            if max_n - min_n > 1e-6:
                 normalized_noise = (noise_map - min_n) / (max_n - min_n)
            else:
                 normalized_noise = np.zeros_like(noise_map)

            # Apply contrast relative to the base intensity
            background = base_intensity + (normalized_noise - 0.5) * contrast

    elif bg_type == 'composite':
        type1 = params.get('type1', 'flat') # Get randomized value
        type2 = params.get('type2', 'perlin') # Get randomized value
        mode = params.get('mode', 'additive') # Get randomized value
        alpha = params.get('alpha', 0.5)

        logger.debug(f"Generating composite background: {type1} + {type2} (mode: {mode}, alpha: {alpha})")

        # Need to create sub-configs for the recursive calls
        # Get the *already randomized* parameters for the sub-types
        params_all = config.get('parameters', {})
        sub_config1 = {'selected_type': type1, 'parameters': params_all, 'types': type1} # Pass necessary keys
        sub_config2 = {'selected_type': type2, 'parameters': params_all, 'types': type2}

        # Check again for composite nesting (shouldn't happen if config is right)
        if type1 == 'composite' or type2 == 'composite':
            logger.warning("Cannot nest composite backgrounds. Falling back to flat.")
            background.fill(0.2)
        else:
            # Call generate_background recursively
            bg1 = generate_background(sub_config1, size, magnification, rng)
            bg2 = generate_background(sub_config2, size, magnification, rng)

            # Blending logic (same as before)
            if mode == 'additive':
                background = bg1 * alpha + bg2 * (1.0 - alpha)
            elif mode == 'multiplicative':
                background = bg1 * bg2
            elif mode == 'overlay':
                  # Formula: 2ab if base < 0.5; 1 - 2(1-a)(1-b) if base >= 0.5
                  # Where a = blend (bg1), b = base (bg2)
                  # We can implement this element-wise using numpy masks
                  logger.debug("Applying Overlay blend mode.")
                  dark_mask = bg2 < 0.5
                  light_mask = ~dark_mask # bg2 >= 0.5

                  # Calculate for dark areas (Multiply * 2)
                  background[dark_mask] = 2.0 * bg1[dark_mask] * bg2[dark_mask]

                  # Calculate for light areas (Screen inverted * 2, then inverted)
                  background[light_mask] = 1.0 - 2.0 * (1.0 - bg1[light_mask]) * (1.0 - bg2[light_mask])

                  # Optional: Apply alpha blending *after* the overlay effect?
                  # Or should alpha control which layer is base/blend?
                  # Current overlay formula assumes bg1 is blend, bg2 is base.
                  # Let's use alpha to blend between the overlay result and the original base (bg2).
                  # This makes alpha control the *strength* of the overlay effect.
                  background = bg2 * (1.0 - alpha) + background * alpha
            else:
                logger.warning(f"Unknown composite mode '{mode}'. Using additive.")
                background = bg1 * alpha + bg2 * (1.0 - alpha)


    else:
        logger.warning(f"Unknown background type '{bg_type}'. Using flat.")
        background.fill(0.2) # Default fallback

    # Ensure background is clipped to [0, 1]
    background = np.clip(background, 0.0, 1.0)
    logger.info(f"Generated '{bg_type}' background.")
    return background.astype(np.float32)

