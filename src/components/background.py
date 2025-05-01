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


def generate_background(config, size, magnification):
    """Generates the background canvas."""
    bg_type = config.get('selected_type', 'flat')
    params = config.get('parameters', {}).get(bg_type, {})
    height, width = size

    if bg_type == 'flat':
        intensity = params.get('intensity', 0.2) # Use randomized value from config
        background = np.full((height, width), intensity, dtype=np.float32)

    elif bg_type == 'gradient':
        start_intensity = params.get('start_intensity', 0.1)
        end_intensity = params.get('end_intensity', 0.3)
        angle = params.get('angle', 0) # Degrees
        grad_type = params.get('type', 'linear')

        background = np.zeros((height, width), dtype=np.float32)
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)

        if grad_type == 'linear':
            rad_angle = np.deg2rad(angle)
            cos_a, sin_a = np.cos(rad_angle), np.sin(rad_angle)
            # Project coordinates onto the gradient direction
            projection = xx * cos_a + yy * sin_a
            # Scale projection to 0-1 range
            min_p, max_p = np.min(projection), np.max(projection)
            if max_p - min_p < 1e-6:
                gradient = np.zeros_like(projection)
            else:
                 gradient = (projection - min_p) / (max_p - min_p)
            background = start_intensity + (end_intensity - start_intensity) * gradient
        elif grad_type == 'radial':
            center_x, center_y = 0.5, 0.5 # Assume center for now
            dist = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
            max_dist = np.sqrt(0.5**2 + 0.5**2) # Dist to corner
            gradient = dist / max_dist
            background = start_intensity + (end_intensity - start_intensity) * gradient
        else:
            print(f"WARNING: Unknown gradient type: {grad_type}. Using flat.")
            background = np.full((height, width), start_intensity, dtype=np.float32)


    elif bg_type == 'perlin':
         if not HAS_PERLIN:
             print("WARNING: Perlin noise requested but library not found. Using flat.")
             intensity = params.get('intensity_range', [0.1, 0.3]) # Fallback
             intensity = intensity[0] if isinstance(intensity, list) else intensity
             background = np.full((height, width), intensity , dtype=np.float32)
         else:
            octaves = params.get('octaves', 6)
            persistence = params.get('persistence', 0.5) # Not directly used by perlin-noise lib
            lacunarity = params.get('lacunarity', 2.0)   # Not directly used by perlin-noise lib
            scale = params.get('scale', 100.0) / magnification # Noise frequency scales with mag
            contrast = params.get('contrast', 0.1)
            base_intensity = params.get('base_intensity', 0.2) # Add a base gray level

            # PerlinNoise library expects octaves. Seed needs to be int.
            noise_gen = PerlinNoise(octaves=octaves, seed=np.random.randint(0, 10000)) # Use sample RNG seed here!

            background = np.zeros((height, width), dtype=np.float32)
            for i in range(height):
                for j in range(width):
                    # Scale coordinates for noise frequency control
                    background[i][j] = noise_gen([i / scale, j / scale])

            # Normalize noise approx -1 to 1 -> 0 to 1, apply contrast and base
            background = (background - np.min(background)) / (np.max(background) - np.min(background) + 1e-6)
            background = base_intensity + (background - 0.5) * contrast
    else:
        print(f"WARNING: Unknown background type '{bg_type}'. Using flat.")
        background = np.full((height, width), 0.2, dtype=np.float32) # Default fallback

    # Ensure background is clipped to [0, 1]
    background = np.clip(background, 0.0, 1.0)
    logger.info(f"Generated '{bg_type}' background.")
    return background.astype(np.float32)
