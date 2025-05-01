import argparse
import os
import random
import logging
from pathlib import Path
from ..src.core.configuration import load_config, override_config
from ..src.core.generator import generate_sample
from ..src.core.utils import get_rng, ensure_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic SEM Image Generator CLI")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to base JSON configuration file")
    parser.add_argument("-o", "--output-dir", type=str, help="Output directory (overrides config)")
    parser.add_argument("-n", "--num-samples", type=int, help="Number of samples to generate (overrides config)")
    parser.add_argument("-s", "--seed", type=int, help="Master random seed (overrides config)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging (overrides config)")
    parser.add_argument("--show-config", action="store_true", help="Print final base config and exit")
    return parser.parse_args()

def run_cli():
    args = parse_args()

    # --- Logging Setup ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger = logging.getLogger(__name__) # Get logger for this module

    # --- Load Base Config ---
    try:
        base_config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    # --- Apply CLI Overrides ---
    overrides = {
        'run_settings.output_dir': args.output_dir,
        'run_settings.num_samples': args.num_samples,
        'run_settings.master_seed': args.seed,
        'run_settings.verbose': args.verbose or base_config.get('run_settings',{}).get('verbose', False), # Prioritize CLI flag
    }
    # Need a function to handle nested overrides properly
    # Simple override for now, assuming flat keys in 'overrides' map to potentially nested keys in base_config
    # Let's refine override_config or use a helper for nested keys
    def set_nested(d, keys, value):
         keys_list = keys.split('.')
         for i, key in enumerate(keys_list[:-1]):
            d = d.setdefault(key, {}) # Create dict if not exists
            if not isinstance(d, dict):
                logger.warning(f"Cannot set nested key '{keys}', intermediate '{key}' is not a dict.")
                return False
         if value is not None:
              logger.info(f"Overriding config '{keys}' -> {value}")
              d[keys_list[-1]] = value
              return True
         return False

    effective_config = base_config.copy() # Start with base
    for key, value in overrides.items():
        set_nested(effective_config, key, value)

    # Update log level based on final effective config
    log_level = logging.DEBUG if effective_config.get('run_settings',{}).get('verbose') else logging.INFO
    logging.getLogger().setLevel(log_level) # Set level on root logger

    if args.show_config:
         import json
         print(json.dumps(effective_config, indent=4))
         return

    # --- Get Final Run Settings ---
    run_settings = effective_config.get('run_settings', {})
    num_samples = run_settings.get('num_samples', 1)
    output_dir = run_settings.get('output_dir', './output')
    master_seed = run_settings.get('master_seed', None)

    # --- Initialize Master RNG ---
    if master_seed is None:
        master_seed = random.randint(0, 2**32 - 1)
        logger.info(f"Generated master seed: {master_seed}")
    else:
        logger.info(f"Using provided master seed: {master_seed}")
    master_rng = get_rng(master_seed)

    # --- Ensure Output Directory Exists ---
    ensure_dir(output_dir)

    # --- Generation Loop ---
    logger.info(f"Starting generation of {num_samples} samples to '{output_dir}'...")
    success_count = 0
    for i in range(num_samples):
        try:
            if generate_sample(i, effective_config, master_rng, output_dir):
                success_count += 1
        except Exception as e:
            logger.error(f"!!! CRITICAL ERROR generating sample {i} !!! : {e}", exc_info=True)
            # Option: stop or continue? Continue for now.

    logger.info(f"Generation finished. {success_count}/{num_samples} samples generated successfully.")
