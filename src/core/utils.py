import numpy as np
import cv2
import os
import random
import json
import logging
import imageio
from pathlib import Path
from scipy.ndimage import map_coordinates

logger = logging.getLogger(__name__)

def get_rng(seed=None):
    """Gets a random number generator instance."""
    return random.Random(seed)

def parse_value(value, rng):
    """Parses a config value, handling ranges and choices."""
    if isinstance(value, list) and len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
        # Assume range [min, max]
        if isinstance(value[0], int) and isinstance(value[1], int):
            return rng.randint(value[0], value[1])
        else:
            return rng.uniform(value[0], value[1])
    elif isinstance(value, list):
        # Assume choice
        if not value: # Check if the list is EMPTY
             logger.warning("Attempted to choose from an empty list. Returning None.")
             return None # Return None if list is empty
        return rng.choice(value) # Choose one item
    else:
        # Assume fixed value
        return value

def ensure_dir(path):
    """Ensures a directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

def normalize_image(image, target_min=0.0, target_max=1.0):
    """Normalizes image to a target range."""
    min_val, max_val = np.min(image), np.max(image)
    if max_val - min_val < 1e-6:
        # Avoid division by zero for flat images
        return np.full(image.shape, target_min, dtype=np.float32)
    normalized = (image - min_val) / (max_val - min_val)
    return normalized * (target_max - target_min) + target_min

def image_to_bit_depth(image, bit_depth):
    """Converts a normalized float image (0-1) to target bit depth."""
    if bit_depth == 8:
        max_val = 255
        dtype = np.uint8
    elif bit_depth == 16:
        max_val = 65535
        dtype = np.uint16
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")
    # Clip to ensure values are within [0, 1] before scaling
    image_clipped = np.clip(image, 0.0, 1.0)
    return (image_clipped * max_val).astype(dtype)

def draw_shapes_on_mask(mask, shapes, color=1):
    """Draws multiple shapes onto a binary mask."""
    # Placeholder: Implement actual shape drawing using cv2
    # Example for circles:
    # for shape in shapes:
    #     if shape['type'] == 'circle':
    #         cv2.circle(mask, (int(shape['center_x']), int(shape['center_y'])),
    #                    int(shape['radius']), color, -1) # -1 for filled
    pass # Replace with actual drawing logic based on shape type

def scale_coords(coords, scale_factor):
    """Scales a list of (x, y) coordinates."""
    return [(int(x * scale_factor), int(y * scale_factor)) for x, y in coords]

# --- Add more utils as needed: color mapping, overlay drawing ---
def get_distinct_colors(n):
    """Generates N visually distinct colors."""
    # Simple approach, cycles through hues
    colors = []
    for i in range(n):
        # PROBLEM: Hue calculation can exceed 180 for large 'n' or certain 'i' values
        # OpenCV HSV hue range is typically 0-179 for uint8 representation
        hue = i * (360.0 / n) # This calculates hue in degrees (0-360)

        # --- Fix Here ---
        # Convert hue to OpenCV's 0-179 range for uint8 HSV
        opencv_hue = int(round((hue / 360.0) * 180.0)) % 180 # Modulo 180 ensures wrap-around

        # Create HSV color (ensure S and V are within uint8 range 0-255)
        # Using 255 for S and V gives bright, saturated colors
        hsv_color = np.uint8([[[opencv_hue, 255, 255]]]) # <-- Use opencv_hue

        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, bgr_color))) # Convert numpy array elements to standard ints for tuples
    return colors


def create_color_visualization(mask, colormap):
    """Creates a colored visualization from an ID mask."""
    if mask.ndim == 3 and mask.shape[2] == 3: # Already color
        return mask
    if not colormap:
         return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # Simple grayscale if no map

    output_vis = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    ids = np.unique(mask)
    for i, obj_id in enumerate(ids):
        if obj_id == 0: continue # Skip background
        color = colormap.get(obj_id, (255, 255, 255)) # Default white if ID not in map
        output_vis[mask == obj_id] = color
    return output_vis

def get_kernel(size, sigma_x, sigma_y=None, angle=0):
    """Creates a 2D Gaussian kernel, potentially rotated."""
    if size % 2 == 0: size += 1 # Ensure odd size
    if sigma_y is None: sigma_y = sigma_x

    center = size // 2
    x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)

    # Rotate coordinates
    angle_rad = -np.deg2rad(angle) # Negative angle for coordinate rotation
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a

    # Calculate Gaussian
    sigma_x2 = 2 * sigma_x**2 + 1e-6 # Add epsilon for stability
    sigma_y2 = 2 * sigma_y**2 + 1e-6
    exponent = (x_rot**2 / sigma_x2) + (y_rot**2 / sigma_y2)
    kernel = np.exp(-exponent)

    # Normalize
    kernel /= kernel.sum()
    return kernel

def visualize_warp_field(warp_field):
    """Visualizes a 2D displacement field (HxWx2) using HSV color space."""
    h, w, _ = warp_field.shape
    dy, dx = warp_field[:, :, 0], warp_field[:, :, 1]

    magnitude = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx) # Angle in radians [-pi, pi]

    hsv = np.zeros((h, w, 3), dtype=np.float32)
    # Angle -> Hue (0-360)
    hsv[:, :, 0] = (angle + np.pi) / (2 * np.pi) * 360
    # Magnitude -> Saturation (normalized)
    max_mag = np.max(magnitude)
    if max_mag > 1e-6:
         hsv[:, :, 1] = magnitude / max_mag
    else:
         hsv[:, :, 1] = 0
    # Value (brightness) = constant 1.0
    hsv[:, :, 2] = 1.0

    vis_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return (vis_bgr * 255).astype(np.uint8)
