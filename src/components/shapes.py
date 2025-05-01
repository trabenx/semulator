import numpy as np
import cv2
import math
from ..core.utils import get_rng # Needed for wavy, irregular polygons

# --- Shape Drawing Functions ---

def draw_circle(canvas, center, radius, color, thickness=-1, anti_aliasing=False):
    line_type = cv2.LINE_AA if anti_aliasing and thickness != -1 else cv2.LINE_8
    # For filled shapes, AA doesn't work directly, might need oversampling or blur after
    cv2.circle(canvas, center, int(radius), color, thickness, line_type)

def draw_rectangle(canvas, pt1, pt2, color, thickness=-1, anti_aliasing=False):
    line_type = cv2.LINE_AA if anti_aliasing and thickness != -1 else cv2.LINE_8
    cv2.rectangle(canvas, pt1, pt2, color, thickness, line_type)

def draw_ellipse(canvas, center, axes, angle, color, thickness=-1, anti_aliasing=False):
    line_type = cv2.LINE_AA if anti_aliasing and thickness != -1 else cv2.LINE_8
    # Ensure axes are integers
    axes = (int(axes[0]), int(axes[1]))
    # OpenCV ellipse axes are typically half-lengths
    cv2.ellipse(canvas, center, axes, angle, 0, 360, color, thickness, line_type)

def draw_rounded_rectangle(canvas, pt1, pt2, corner_radius, color, thickness=-1):
    """Draws a rounded rectangle. Anti-aliasing is approximate."""
    # Based on StackOverflow answers - drawing arcs and lines
    x1, y1 = pt1
    x2, y2 = pt2
    r = int(corner_radius)
    if r <= 0: # Just draw a normal rectangle
        cv2.rectangle(canvas, pt1, pt2, color, thickness)
        return

    if thickness == -1: # Filled
        # Draw rectangles for the center part
        cv2.rectangle(canvas, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(canvas, (x1, y1 + r), (x2, y2 - r), color, -1)
        # Draw circles for the corners
        cv2.circle(canvas, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(canvas, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(canvas, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(canvas, (x2 - r, y2 - r), r, color, -1)
    else: # Outline
        # Arcs for corners
        cv2.ellipse(canvas, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(canvas, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(canvas, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(canvas, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
        # Lines for sides
        cv2.line(canvas, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(canvas, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(canvas, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(canvas, (x2, y1 + r), (x2, y2 - r), color, thickness)

def draw_line(canvas, pt1, pt2, color, thickness=1, anti_aliasing=False):
    line_type = cv2.LINE_AA if anti_aliasing else cv2.LINE_8
    # Ensure thickness is at least 1
    thickness = max(1, int(thickness))
    cv2.line(canvas, pt1, pt2, color, thickness, line_type)

def draw_wavy_line(canvas, pt1, pt2, amplitude, wavelength, color, thickness=1, anti_aliasing=False, rng=None):
    if rng is None: rng = np.random.RandomState() # Use numpy's default RNG if none provided

    x1, y1 = pt1
    x2, y2 = pt2
    line_type = cv2.LINE_AA if anti_aliasing else cv2.LINE_8
    thickness = max(1, int(thickness))

    dx, dy = x2 - x1, y2 - y1
    line_length = np.sqrt(dx**2 + dy**2)
    if line_length < 1e-6: return # Avoid division by zero

    angle = np.arctan2(dy, dx)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    perp_cos_a, perp_sin_a = -sin_a, cos_a # Perpendicular direction

    num_points = max(10, int(line_length / (wavelength / 10))) # Number of segments
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        base_x = x1 + t * dx
        base_y = y1 + t * dy

        # Calculate offset based on sine wave along the line
        offset = amplitude * np.sin(t * line_length / wavelength * 2 * np.pi + rng.uniform(0, 2*np.pi)) # Add phase shift

        # Apply offset perpendicular to the line direction
        point_x = int(base_x + offset * perp_cos_a)
        point_y = int(base_y + offset * perp_sin_a)
        points.append((point_x, point_y))

    if points:
        cv2.polylines(canvas, [np.array(points)], isClosed=False, color=color, thickness=thickness, lineType=line_type)


def draw_polygon(canvas, vertices, color, thickness=-1, anti_aliasing=False):
    line_type = cv2.LINE_AA if anti_aliasing and thickness != -1 else cv2.LINE_8
    pts = np.array(vertices, dtype=np.int32)
    if thickness == -1:
        cv2.fillPoly(canvas, [pts], color, line_type) # line_type might apply AA to edges
    else:
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness, lineType=line_type)

def create_regular_polygon_vertices(center, radius, num_sides, irregularity=0.0, rng=None):
    """Generates vertices for a regular or irregular polygon."""
    if rng is None: rng = np.random.RandomState()
    cx, cy = center
    vertices = []
    angle_step = 2 * np.pi / num_sides
    for i in range(num_sides):
        angle = i * angle_step
        # Add irregularity to radius
        current_radius = radius * (1 + rng.uniform(-irregularity, irregularity))
        x = cx + current_radius * np.cos(angle)
        y = cy + current_radius * np.sin(angle)
        vertices.append((int(x), int(y)))
    return vertices

# --- Predefined Shapes (Example H shape) ---
# Define vertices relative to a center (0,0) and scale
PREDEFINED_SHAPES = {
    # Format: [(x1,y1), (x2,y2), ...] clockwise or counter-clockwise
    'H': [(-1,-1), (-1,-0.2), (-0.2,-0.2), (-0.2,-1), (0.2,-1), (0.2,-0.2),
            (1,-0.2), (1,1), (0.2,1), (0.2,0.2), (-0.2,0.2), (-0.2,1)],
    'T': [(-1,-1), (1,-1), (1,-0.6), (0.2,-0.6), (0.2,1), (-0.2,1), (-0.2,-0.6), (-1,-0.6)],
    'L': [(-1,-1), (-1,1), (-0.6,1), (-0.6,-0.6), (1,-0.6), (1,-1)],
    'E': [(-1,-1), (1,-1), (1,-0.6), (-0.6,-0.6), (-0.6,-0.2), (0.6,-0.2), (0.6,0.2),
            (-0.6,0.2), (-0.6,0.6), (1,0.6), (1,1), (-1,1)],
    'I': [(-0.6,-1), (0.6,-1), (0.6,-0.8), (0.2,-0.8), (0.2,0.8), (0.6,0.8),
            (0.6,1), (-0.6,1), (-0.6,0.8), (-0.2,0.8), (-0.2,-0.8), (-0.6,-0.8)],
    'C': [ (1,-1), (1,1), (-0.6,1), (-0.6,0.6), (0.6,0.6), (0.6,-0.6),
            (-0.6,-0.6), (-0.6,-1)], # Open on the left
    'U': [(-1,1), (-1,-1), (1,-1), (1,1), (0.6,1), (0.6,-0.6), (-0.6,-0.6), (-0.6,1)],
    'X': [(-1,-0.6), (-0.6,-1), (-0.2,-1), (0,-0.2), (0.2,-1), (0.6,-1), (1,-0.6),
            (0.2,0), (1,0.6), (0.6,1), (0.2,1), (0,0.2), (-0.2,1), (-0.6,1),
            (-1,0.6), (-0.2,0)],
    'Y': [(-1,-1), (-0.6,-1), (0,-0.2), (0.6,-1), (1,-1), (0.2,0), (0.2,1),
            (-0.2,1), (-0.2,0)],
    # Add more complex shapes if needed
}

def get_predefined_shape_vertices(shape_key, center, size, rotation_deg=0):
     """Gets scaled and rotated vertices for predefined shapes."""
     if shape_key not in PREDEFINED_SHAPES:
         raise ValueError(f"Unknown predefined shape key: {shape_key}")

     base_vertices = np.array(PREDEFINED_SHAPES[shape_key], dtype=float)
     # Scale vertices - use size as scaling factor for the [-1, 1] coordinates
     scaled_vertices = base_vertices * (size / 2.0)

     # Rotate vertices
     if abs(rotation_deg % 360) > 1e-3:
          angle_rad = np.deg2rad(rotation_deg)
          cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
          rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
          rotated_vertices = scaled_vertices @ rotation_matrix.T # Apply rotation
     else:
          rotated_vertices = scaled_vertices

     # Translate to center
     final_vertices = rotated_vertices + np.array(center)

     return [(int(v[0]), int(v[1])) for v in final_vertices]



# --- Factory Function ---
def draw_shape(canvas, shape_type, params, color, thickness=-1, anti_aliasing=False, rng=None):
    """Factory to call the correct drawing function."""
    center_pos = (int(params.get('center_x', canvas.shape[1]//2)), int(params.get('center_y', canvas.shape[0]//2)))
    rotation = params.get('rotation', 0) # Angle in degrees

    if shape_type == 'circle':
        radius = params.get('radius', 10)
        draw_circle(canvas, center_pos, int(radius), color, thickness, anti_aliasing) # Ensure radius is int
    elif shape_type == 'rectangle':
        w = params.get('width', 20)
        h = params.get('height', w) # Default to square
        # Draw rotated rectangle if angle != 0
        if abs(rotation % 360) > 1e-3:
             box = cv2.boxPoints(((center_pos[0], center_pos[1]), (w, h), rotation))
             draw_polygon(canvas, box, color, thickness, anti_aliasing)
        else: # Axis aligned
             pt1 = (center_pos[0] - w // 2, center_pos[1] - h // 2)
             pt2 = (center_pos[0] + w // 2, center_pos[1] + h // 2)
             draw_rectangle(canvas, pt1, pt2, color, thickness, anti_aliasing)
    elif shape_type == 'ellipse':
         axes = (params.get('width', 20) // 2, params.get('height', 15) // 2)
         draw_ellipse(canvas, center_pos, axes, rotation, color, thickness, anti_aliasing)
    elif shape_type == 'rounded_rectangle':
         w = params.get('width', 20)
         h = params.get('height', w)
         corner_radius_fraction = params.get('corner_radius_fraction', 0.2)
         corner_radius = min(w, h) / 2 * corner_radius_fraction
         pt1 = (center_pos[0] - w // 2, center_pos[1] - h // 2)
         pt2 = (center_pos[0] + w // 2, center_pos[1] + h // 2)
         if abs(rotation % 360) > 1e-3:
             print("Warning: Rotation not fully implemented for rounded_rectangle. Drawing axis-aligned.")
         draw_rounded_rectangle(canvas, pt1, pt2, corner_radius, color, thickness)
    elif shape_type == 'line':
         pt1 = (int(params['x1']), int(params['y1']))
         pt2 = (int(params['x2']), int(params['y2']))
         line_thickness = max(1, int(params.get('thickness', thickness)))
         draw_line(canvas, pt1, pt2, color, line_thickness, anti_aliasing)
    elif shape_type == 'wavy_line':
         pt1 = (int(params['x1']), int(params['y1']))
         pt2 = (int(params['x2']), int(params['y2']))
         amplitude = params.get('amplitude', 5)
         wavelength = params.get('wavelength', 30)
         line_thickness = max(1, int(params.get('thickness', thickness)))
         draw_wavy_line(canvas, pt1, pt2, amplitude, wavelength, color, line_thickness, anti_aliasing, rng)
    elif shape_type == 'polygon':
         radius = params.get('radius', 20)
         num_sides = params.get('num_sides', 6)
         irregularity = params.get('irregularity', 0.0)
         vertices = create_regular_polygon_vertices(center_pos, radius, num_sides, irregularity, rng)
         if abs(rotation % 360) > 1e-3:
              center_np = np.array(center_pos)
              vertices_np = np.array(vertices) - center_np
              angle_rad = np.deg2rad(rotation)
              cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
              rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
              rotated_vertices = vertices_np @ rotation_matrix.T + center_np
              vertices = [(int(v[0]), int(v[1])) for v in rotated_vertices]
         draw_polygon(canvas, vertices, color, thickness, anti_aliasing)
    elif shape_type in PREDEFINED_SHAPES:
         size = params.get('size', 30) # Define size for predefined shape
         vertices = get_predefined_shape_vertices(shape_type, center_pos, size, rotation)
         draw_polygon(canvas, vertices, color, thickness, anti_aliasing)
    else:
        print(f"Warning: Shape type '{shape_type}' not implemented for drawing.")




def create_shape_mask(shape_type, params, size, anti_aliasing=False, rng=None):
    """Creates a binary mask for a single shape."""
    mask = np.zeros(size, dtype=np.uint8)
    # Masks should always be binary (no AA), filled
    draw_shape(mask, shape_type, params, color=1, thickness=-1, anti_aliasing=False, rng=rng)
    return mask

def render_shape(shape_type, params, size, intensity, anti_aliasing=True, rng=None):
     """Renders a single shape with intensity."""
     render = np.zeros(size, dtype=np.float32)
     # Draw filled shape with intensity 1.0, potentially with AA
     # For filled AA, we might draw binary mask then blur slightly
     if anti_aliasing:
         # Draw binary first
         mask = np.zeros(size, dtype=np.float32)
         draw_shape(mask, shape_type, params, color=1.0, thickness=-1, anti_aliasing=False, rng=rng)
         # Apply subtle blur for anti-aliasing effect
         render = cv2.GaussianBlur(mask, (3, 3), 0.5)
     else:
         draw_shape(render, shape_type, params, color=1.0, thickness=-1, anti_aliasing=False, rng=rng)

     render *= intensity
     return render
