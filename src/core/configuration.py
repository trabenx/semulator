import json
import logging
from .utils import parse_value, get_rng

logger = logging.getLogger(__name__)

def load_config(config_path):
    """Loads base JSON configuration."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from: {config_path}")
        raise

def override_config(config, overrides):
    """Overrides config dict with values from another dict."""
    for key, value in overrides.items():
        if value is not None:
            # Simple override, assumes flat structure for overrides for now
            # For nested overrides, a recursive merge function would be needed
            if key in config:
                 logger.debug(f"Overriding config '{key}': {config[key]} -> {value}")
                 config[key] = value
            elif '.' in key: # Handle nested keys like "image_settings.resolution"
                parts = key.split('.')
                d = config
                try:
                    for i, part in enumerate(parts[:-1]):
                        d = d[part]
                    original_value = d.get(parts[-1], 'N/A')
                    d[parts[-1]] = value
                    logger.debug(f"Overriding nested config '{key}': {original_value} -> {value}")
                except KeyError:
                    logger.warning(f"Cannot override nested key '{key}': intermediate key not found.")
            else:
                 logger.warning(f"Override key '{key}' not found in base config.")
    return config


def _recursive_randomize(data, rng):
    """Recursively traverses config dict/list to randomize values."""
    if isinstance(data, dict):
        # Check for special randomization keys first (e.g., ranges)
        keys = list(data.keys())
        processed_as_param = False
        for key in keys:
             if key.endswith(('_range', '_choices', '_probability')):
                 # If a key signals a parameter range/choice, process its parent dict
                 # This is tricky. Let's simplify: assume parse_value handles lists correctly.
                 pass

        # If not processed as a special parameter dict, recurse
        if not processed_as_param:
            return {k: _recursive_randomize(v, rng) for k, v in data.items()}
        else:
             return data # Stop recursion if handled as parameter
    elif isinstance(data, list):
         # If it looks like a parameter list (range or choices), parse it
         if len(data) >= 1 and isinstance(data[0], (int, float, str)): # Heuristic
             parsed = parse_value(data, rng)
             # logger.debug(f"Randomized list {data} -> {parsed}")
             return parsed
         else: # Otherwise, assume it's a list of structures to recurse into
             return [_recursive_randomize(item, rng) for item in data]

    # Check if the value itself needs randomization (e.g., simple list choice)
    # This is already handled by parse_value if called on the list directly.
    # Let's ensure parse_value is called at the right points.

    # Re-thinking the recursion: It might be cleaner to identify parameter patterns
    # by key names (_range, _choices, etc.) at the parent level.

    # Simpler approach: Apply parse_value to leaf nodes that are lists.
    if isinstance(data, list):
        # Only parse if it looks like a value list, not a list of dicts
        if len(data) > 0 and not isinstance(data[0], (dict, list)):
            return parse_value(data, rng)
        else:
            # List of complex items, recurse
            return [_recursive_randomize(item, rng) for item in data]
    elif isinstance(data, dict):
        return {k: _recursive_randomize(v, rng) for k, v in data.items()}
    else:
         return data # Leaf node, no randomization needed unless it was a list handled above

def randomize_config_for_sample(base_config, sample_seed):
    """Randomizes a configuration dictionary for a single sample."""
    rng = get_rng(sample_seed)
    randomized_config = {}

    # Define sections that need full recursive randomization
    sections_to_randomize = ['background', 'layering', 'artifact_raffle']

    for key, value in base_config.items():
        if key in sections_to_randomize:
            # logger.debug(f"Randomizing section: {key}")
            randomized_config[key] = _recursive_randomize(value, rng)
        else:
            # Copy other sections directly (like run_settings, image_settings)
            # Although some parameters might need randomization here too (e.g., magnification?)
            # Let's handle specific cases like magnification if needed
            if key == 'image_settings':
               randomized_config[key] = _recursive_randomize(value, rng) # Randomize within image settings too
            else:
               randomized_config[key] = value # Assume fixed or handled elsewhere

    # Special handling for selecting layers based on probability
    if 'layering' in randomized_config and 'layers' in randomized_config['layering']:
        available_layers = randomized_config['layering']['layers']
        num_layers_to_gen = parse_value(randomized_config['layering']['num_layers_range'], rng)
        selected_layers = []
        # Weighted random choice (or just use probabilities as filters if sum > 1?)
        # Simple approach: Pick num_layers_to_gen randomly respecting probability somewhat
        candidates = [layer for layer in available_layers if rng.random() < layer.get('probability', 1.0)]
        if len(candidates) >= num_layers_to_gen:
             selected_layers = rng.sample(candidates, num_layers_to_gen)
        else:
             # If not enough candidates pass probability, take all that passed + fill randomly
             selected_layers = candidates
             needed = num_layers_to_gen - len(selected_layers)
             remaining = [layer for layer in available_layers if layer not in selected_layers]
             if needed > 0 and remaining:
                 selected_layers.extend(rng.sample(remaining, min(needed, len(remaining))))

        randomized_config['layering']['selected_layers'] = selected_layers
        # logger.debug(f"Selected {len(selected_layers)} layers: {[l['layer_id'] for l in selected_layers]}")


    # Select background type
    if 'background' in randomized_config:
        bg_type = parse_value(randomized_config['background']['types'], rng)
        randomized_config['background']['selected_type'] = bg_type
        # Keep only params for the selected type? Or keep all randomized? Keep all for now.
        # logger.debug(f"Selected background type: {bg_type}")


    # logger.info(f"Generated randomized config for seed {sample_seed}")
    return randomized_config