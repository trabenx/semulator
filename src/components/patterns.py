import numpy as np
import random
import math

def _get_shape_dimensions(params):
    """Helper to estimate main dimensions (width, height) from shape params."""
    radius = params.get('radius')
    width = params.get('width')
    height = params.get('height')

    if radius:
        return radius * 2, radius * 2
    if width and height:
        return width, height
    if width:
        return width, width # Assume square if only width
    if height:
        return height, height # Assume square if only height
    return 20, 20 # Default fallback estimate

def generate_single_position(size, params):
    h, w = size
    cx = params.get('center_x', w // 2)
    cy = params.get('center_y', h // 2)
    return [(int(cx), int(cy))]

def generate_grid_positions(size, params, rng):
    h, w = size
    rows = params.get('rows', 5)
    cols = params.get('cols', 5)
    spacing_factor = params.get('spacing_factor', 2.0)
    jitter_fraction = params.get('jitter_fraction', 0.0)
    shape_w, shape_h = _get_shape_dimensions(params)

    spacing_x = spacing_factor * shape_w
    spacing_y = spacing_factor * shape_h
    grid_width = (cols - 1) * spacing_x
    grid_height = (rows - 1) * spacing_y
    start_x = (w - grid_width) / 2
    start_y = (h - grid_height) / 2

    positions = []
    for r in range(rows):
        for c in range(cols):
            base_x = start_x + c * spacing_x
            base_y = start_y + r * spacing_y
            jitter_x = rng.uniform(-0.5, 0.5) * jitter_fraction * spacing_x
            jitter_y = rng.uniform(-0.5, 0.5) * jitter_fraction * spacing_y
            x = base_x + jitter_x
            y = base_y + jitter_y
            if 0 <= x < w and 0 <= y < h:
                 positions.append((int(x), int(y)))
    return positions

def generate_hex_grid_positions(size, params, rng):
    h, w = size
    # Estimate rows/cols based on size and spacing if not provided? More complex. Assume provided.
    # Or determine max possible rows/cols based on spacing.
    spacing_factor = params.get('spacing_factor', 2.0)
    jitter_fraction = params.get('jitter_fraction', 0.0)
    shape_w, shape_h = _get_shape_dimensions(params) # Use dimensions for spacing

    # Hex grid spacing: dy = sqrt(3)/2 * dx approx
    spacing_x = spacing_factor * shape_w
    spacing_y = spacing_factor * shape_h * (np.sqrt(3) / 2.0)

    # Calculate number of rows/cols that fit (approx)
    cols_fit = int(w / spacing_x) + 1
    rows_fit = int(h / spacing_y) + 1

    grid_width = (cols_fit -1) * spacing_x
    grid_height = (rows_fit -1) * spacing_y
    start_x = (w - grid_width) / 2
    start_y = (h - grid_height) / 2


    positions = []
    for r in range(rows_fit):
        for c in range(cols_fit):
            # Offset every second row
            offset_x = 0.5 * spacing_x if r % 2 != 0 else 0.0
            base_x = start_x + c * spacing_x + offset_x
            base_y = start_y + r * spacing_y

            jitter_x = rng.uniform(-0.5, 0.5) * jitter_fraction * spacing_x
            jitter_y = rng.uniform(-0.5, 0.5) * jitter_fraction * spacing_y
            x = base_x + jitter_x
            y = base_y + jitter_y

            if 0 <= x < w and 0 <= y < h:
                positions.append((int(x), int(y)))
    return positions


def generate_radial_grid_positions(size, params, rng):
    h, w = size
    center_x, center_y = w // 2, h // 2
    num_rings = params.get('rings', 3)
    shapes_per_ring_base = params.get('shapes_per_ring', 6) # Can scale with radius
    radius_step = params.get('radius_step', 50)
    jitter_fraction = params.get('jitter_fraction', 0.0) # Jitter in radius and angle

    positions = []
    for r_idx in range(1, num_rings + 1): # Start from ring 1
        radius = r_idx * radius_step
        # Increase shapes per ring further out?
        num_shapes = shapes_per_ring_base # * r_idx # Optional scaling

        for s_idx in range(num_shapes):
            base_angle = (s_idx / num_shapes) * 2 * np.pi

            # Add jitter
            angle_jitter = rng.uniform(-0.5, 0.5) * jitter_fraction * (2 * np.pi / num_shapes)
            radius_jitter = rng.uniform(-0.5, 0.5) * jitter_fraction * radius_step

            current_angle = base_angle + angle_jitter
            current_radius = radius + radius_jitter

            x = center_x + current_radius * np.cos(current_angle)
            y = center_y + current_radius * np.sin(current_angle)

            if 0 <= x < w and 0 <= y < h:
                positions.append((int(x), int(y)))
    return positions


def generate_random_scatter_positions(size, params, rng):
    h, w = size
    num_shapes = params.get('num_shapes', 20)
    positions = []
    for _ in range(num_shapes):
        x = rng.uniform(0, w)
        y = rng.uniform(0, h)
        positions.append((int(x), int(y)))
    return positions

def generate_full_span_vertical_positions(size, params, rng):
    h, w = size
    count = params.get('count', 10)
    spacing_jitter_fraction = params.get('spacing_jitter_fraction', 0.1)

    avg_spacing = w / (count + 1)
    positions = []
    current_x = 0
    for i in range(count):
         spacing = avg_spacing * (1 + rng.uniform(-spacing_jitter_fraction, spacing_jitter_fraction))
         current_x += spacing
         x_pos = int(current_x)
         if 0 <= x_pos < w:
             pt1 = (x_pos, 0)
             pt2 = (x_pos, h - 1)
             positions.append((pt1, pt2))
    return positions

def generate_full_span_horizontal_positions(size, params, rng):
    h, w = size
    count = params.get('count', 10)
    spacing_jitter_fraction = params.get('spacing_jitter_fraction', 0.1)

    avg_spacing = h / (count + 1)
    positions = []
    current_y = 0
    for i in range(count):
         spacing = avg_spacing * (1 + rng.uniform(-spacing_jitter_fraction, spacing_jitter_fraction))
         current_y += spacing
         y_pos = int(current_y)
         if 0 <= y_pos < h:
             pt1 = (0, y_pos)
             pt2 = (w - 1, y_pos)
             positions.append((pt1, pt2))
    return positions

def get_pattern_positions(pattern_type, size, shape_params, pattern_params, rng):
    """Factory to get positions based on pattern type."""
    # Combine params, giving pattern_params precedence for pattern-specific keys
    # Pass shape params through for dimension estimation
    combined_params = shape_params.copy()
    combined_params.update(pattern_params)

    if pattern_type == 'single':
        return generate_single_position(size, combined_params)
    elif pattern_type == 'grid':
        return generate_grid_positions(size, combined_params, rng)
    elif pattern_type == 'hex_grid':
        return generate_hex_grid_positions(size, combined_params, rng)
    elif pattern_type == 'radial_grid':
        return generate_radial_grid_positions(size, combined_params, rng)
    elif pattern_type == 'random_scatter':
        return generate_random_scatter_positions(size, combined_params, rng)
    elif pattern_type == 'full_span_vertical':
        return generate_full_span_vertical_positions(size, combined_params, rng)
    elif pattern_type == 'full_span_horizontal':
        return generate_full_span_horizontal_positions(size, combined_params, rng)
    else:
        print(f"Warning: Pattern type '{pattern_type}' not implemented.")
        return []
