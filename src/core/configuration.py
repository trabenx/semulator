import json
import logging
import copy
from .utils import parse_value, get_rng

logger = logging.getLogger(__name__)

def set_nested(d, keys, value, create_missing=True):
    """
    Sets a value in a nested dictionary using a dot-separated key string.

    Args:
        d (dict): The dictionary to modify.
        keys (str): The dot-separated key (e.g., "a.b.c").
        value: The value to set.
        create_missing (bool): If True, creates intermediate dictionaries if keys don't exist.

    Returns:
        bool: True if the value was set successfully, False otherwise.
    """
    keys_list = keys.split('.')
    current_level = d
    for i, key in enumerate(keys_list[:-1]):
        if key not in current_level:
            if create_missing:
                current_level[key] = {}
            else:
                logger.warning(f"Cannot set nested key '{keys}', intermediate key '{key}' not found.")
                return False
        # Check if it's actually a dictionary before descending
        if not isinstance(current_level.get(key), dict):
             if create_missing:
                 current_level[key] = {} # Overwrite if create_missing is True and it's not a dict
             else:
                logger.warning(f"Cannot set nested key '{keys}', intermediate '{key}' is not a dict (found type: {type(current_level.get(key))}).")
                return False
        current_level = current_level[key]

    final_key = keys_list[-1]
    current_level[final_key] = value
    return True


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
    """
    Recursively traverses config dict/list to randomize values based on key names.
    Randomizes values associated with keys ending in _range, _choices, _probability.
    Strips the suffix from the key after randomization.
    """
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            # --- Revised Suffix Handling ---
            processed = False
            if key.endswith('_range'):
                new_key = key.rsplit('_range', 1)[0]
                # Ensure value is actually a list/tuple suitable for range parsing
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    parsed_value = parse_value(value, rng)
                    new_dict[new_key] = parsed_value
                    processed = True
                else:
                    logger.warning(f"Key '{key}' ends with _range but value is not a 2-element list/tuple: {value}. Keeping original.")
                    new_dict[key] = _recursive_randomize(value, rng) # Recurse into value if structure unknown

            elif key.endswith('_choices'):
                new_key = key.rsplit('_choices', 1)[0]
                # Ensure value is a list suitable for choice parsing
                if isinstance(value, list):
                    parsed_value = parse_value(value, rng) # parse_value handles choice from list
                    new_dict[new_key] = parsed_value # Store the single chosen item
                    processed = True
                else:
                     logger.warning(f"Key '{key}' ends with _choices but value is not a list: {value}. Keeping original.")
                     new_dict[key] = _recursive_randomize(value, rng) # Recurse

            elif key.endswith('_probability'):
                 new_key = key.rsplit('_probability', 1)[0]
                 # Ensure value is suitable for probability parsing (e.g., number or range)
                 try:
                      parsed_value = parse_value(value, rng)
                      # Probabilities are usually floats, ensure it's treated as such if needed downstream
                      new_dict[new_key] = float(parsed_value) if isinstance(parsed_value, (int, float)) else parsed_value
                      processed = True
                 except (TypeError, ValueError):
                      logger.warning(f"Could not parse probability for key '{key}': {value}. Keeping original.")
                      new_dict[key] = _recursive_randomize(value, rng) # Recurse

            # --- End Revised Suffix Handling ---

            if not processed:
                # Key doesn't indicate randomization OR failed validation,
                # so recurse into value if dict/list
                new_dict[key] = _recursive_randomize(value, rng)
        return new_dict
    elif isinstance(data, list):
        # Recurse into list items (important for lists of layers, artifacts, etc.)
        return [_recursive_randomize(item, rng) for item in data]
    else:
        # Leaf node (primitive type), return as is
        return data


def randomize_config_for_sample(base_config, sample_seed):
    """
    Randomizes a configuration dictionary for a single sample using key-based logic.
    Also handles post-processing like selecting layers and background type.
    """
    rng = get_rng(sample_seed)
    logger.debug(f"--- Randomizing config for seed {sample_seed} ---")
    config_copy = copy.deepcopy(base_config)

    # Apply the key-based randomization recursively
    randomized_config = _recursive_randomize(config_copy, rng)

    # --- Post-processing after _recursive_randomize ---

    # Select layers based on the randomized 'probability' in each layer definition
    # and the randomized 'num_layers' value.
    logger.debug(f"--- Finished recursive randomization for seed {sample_seed} ---")


    # Post-processing for layer selection
    if 'layering' in randomized_config and isinstance(randomized_config['layering'], dict):
        # Get the list of all possible layer definitions (already randomized internally)
        all_possible_layers = randomized_config['layering'].get('layers', [])
        num_layers_to_gen = randomized_config['layering'].get('num_layers', 1) # Randomized number

        # --- Filter layers based on the 'enabled' flag ---
        # Default to enabled=True if the flag is missing in the config
        enabled_layers = [
            layer for layer in all_possible_layers
            if layer.get('enabled', True) # Check 'enabled', default to True if missing
        ]
        logger.debug(f"Found {len(enabled_layers)} enabled layers out of {len(all_possible_layers)}.")
        # --- End Filter ---

        # --- Select from ENABLED layers based on probability ---
        selected_layers = []
        # Ensure we only work with the enabled layers pool
        if enabled_layers:
            candidates = [
                layer for layer in enabled_layers
                if rng.random() < layer.get('probability', 1.0) # Check probability
            ]
            logger.debug(f"{len(candidates)} layers passed probability check.")

            # Sample selection logic (using only candidates from enabled layers)
            target_num = min(num_layers_to_gen, len(enabled_layers)) # Can't select more than available enabled

            if len(candidates) >= target_num:
                 selected_layers = rng.sample(candidates, target_num)
            else:
                 # Take all candidates passing probability, fill remaining needed randomly from other enabled layers
                 selected_layers = candidates
                 needed = target_num - len(selected_layers)
                 remaining_enabled = [layer for layer in enabled_layers if layer not in selected_layers]
                 if needed > 0 and remaining_enabled:
                     selected_layers.extend(rng.sample(remaining_enabled, min(needed, len(remaining_enabled))))

        randomized_config['layering']['selected_layers'] = selected_layers
        logger.debug(f"Selected {len(selected_layers)} layers for config seed {sample_seed}.")


    # Optional Sanity Check for resolution (keep if desired)
    if 'image_settings' in randomized_config and not isinstance(randomized_config['image_settings'].get('resolution'), list):
         logger.warning(f"Resolution key 'resolution' in config is not a list after randomization: {randomized_config['image_settings'].get('resolution')}. Check randomization logic.")
         if isinstance(base_config.get('image_settings', {}).get('resolution'), list):
              randomized_config['image_settings']['resolution'] = base_config['image_settings']['resolution']


    # logger.info(f"Generated randomized config for seed {sample_seed}") # Use debug level maybe
    return randomized_config



#    # Store the selected background type.
#    # _recursive_randomize turned 'types_choices' into 'types' holding the chosen string.
#    if 'background' in randomized_config and isinstance(randomized_config['background'], dict):
#        selected_bg_type = randomized_config['background'].get('types') # Get the chosen type
#        if selected_bg_type:
#             randomized_config['background']['selected_type'] = selected_bg_type
#             # logger.debug(f"Selected background type: {selected_bg_type}")
#        # Parameters for all background types were randomized by the recursion.
#
#    # --- Sanity Check (Optional but helpful) ---
#    # Ensure critical fixed values weren't accidentally modified if logic has errors
#    if 'image_settings' in randomized_config and not isinstance(randomized_config['image_settings'].get('resolution'), list):
#         logger.warning(f"Resolution key 'resolution' in config is not a list after randomization: {randomized_config['image_settings'].get('resolution')}. Check randomization logic.")
#         # Attempt to restore from base if possible (might indicate deeper issue)
#         if isinstance(base_config.get('image_settings', {}).get('resolution'), list):
#              randomized_config['image_settings']['resolution'] = base_config['image_settings']['resolution']
#
#
#    # logger.info(f"Generated randomized config for seed {sample_seed}")
#    return randomized_config
