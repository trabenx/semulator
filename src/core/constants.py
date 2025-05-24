# Basic geometric primitives
_BASE_SHAPES = [
    "circle",
    "rectangle",
    "line",
    "rounded_rectangle",
    "ellipse",
    "polygon", # General polygon
    "wavy_line"
    # Add "arc", "path" here if you implement them
]

# Predefined custom shapes (keys from PREDEFINED_SHAPES in shapes.py)
_PREDEFINED_CUSTOM_SHAPES = [
    "H", "T", "L", "E", "I", "C", "U", "X", "Y"
    # Add any other keys you defined in shapes.PREDEFINED_SHAPES
]

# Combine and create the map
SHAPE_TYPE_MAP = {"background": 0}
current_id = 1
for shape_name in _BASE_SHAPES:
    SHAPE_TYPE_MAP[shape_name] = current_id
    current_id += 1

for shape_name in _PREDEFINED_CUSTOM_SHAPES:
    SHAPE_TYPE_MAP[shape_name] = current_id
    current_id += 1

# Number of classes for segmentation model (including background)
NUM_SHAPE_CLASSES = len(SHAPE_TYPE_MAP)
MAX_PREDICTABLE_LAYERS = 5 # Will be set by generator from config

if __name__ == '__main__':
    # Print for verification
    print("SHAPE_TYPE_MAP:")
    for name, id_val in SHAPE_TYPE_MAP.items():
        print(f"  '{name}': {id_val}")
    print(f"\nNUM_SHAPE_CLASSES: {NUM_SHAPE_CLASSES}")
