import argparse
import os
import random
import logging
import imageio
from pathlib import Path

# This script should *not* be run directly.
# It's designed to be imported and run by the top-level 'main.py'.
# 'main.py' sets up the necessary paths and context.
#
# Run the CLI using: python main.py cli [cli arguments...]
#
# We keep the relative imports here because they are correct when run via main.py.
# Adding absolute imports or complex sys.path manipulation here would mask
# the underlying problem (running the script incorrectly) and make the
# structure less maintainable.

from src.core.configuration import load_config
from src.core.generator import generate_sample
from src.core.utils import get_rng, ensure_dir
from src.core.configuration import set_nested # Import the helper used below

# --- CLI Argument Parser ---
# This parser is intended to be used *after* main.py has parsed the 'mode' argument.
def parse_cli_args():
    parser = argparse.ArgumentParser(description="Synthetic SEM Image Generator CLI (run via main.py)")
    # Arguments defined here should *not* include 'mode'
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to base JSON configuration file")
    parser.add_argument("-o", "--output-dir", type=str, help="Output directory (overrides config: run_settings.output_dir)")
    parser.add_argument("-n", "--num-samples", type=int, help="Number of samples to generate (overrides config: run_settings.num_samples)")
    parser.add_argument("-s", "--seed", type=int, help="Master random seed (overrides config: run_settings.master_seed)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging (overrides config: run_settings.verbose)")
    parser.add_argument("--show-config", action="store_true", help="Print final effective config and exit")
    # Add other CLI overrides if needed, using dot notation for help text if desired
    # Example: parser.add_argument("--magnification", type=float, help="Override image_settings.magnification")
    return parser.parse_args() # Parses arguments from sys.argv (which main.py modified)

# --- Main CLI Runner Function ---
def run_cli():
    # Parse arguments specifically for the CLI mode
    args = parse_cli_args()

    # --- Logging Setup ---
    # Base logging config might be set in main.py or here based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    # Configure root logger if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    else: # If already configured (e.g. by main.py), just set the level
         logging.getLogger().setLevel(log_level)

    logger = logging.getLogger(__name__) # Get logger for this module
    logger.info("Running in CLI mode.")

    # --- Load Base Config ---
    try:
        base_config = load_config(args.config)
        logger.info(f"Loaded base configuration from: {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        return # Exit if config fails

    # --- Apply CLI Overrides ---
    # Create the effective_config by copying base_config
    effective_config = {}
    try:
        import json # Use json for a simple deep copy
        effective_config = json.loads(json.dumps(base_config))
        logger.debug("Created deep copy of base config.")
    except Exception as e:
        logger.error(f"Failed to deep copy base config: {e}", exc_info=True)
        effective_config = base_config # Fallback to shallow copy if deep copy fails

    # Define overrides from CLI arguments
    # Map CLI arg destinations to nested config keys
    cli_overrides_map = {
        'output_dir': 'run_settings.output_dir',
        'num_samples': 'run_settings.num_samples',
        'seed': 'run_settings.master_seed',
        'verbose': 'run_settings.verbose',
        # Add mappings for any other direct overrides added to parse_cli_args
        # 'magnification': 'image_settings.magnification',
    }

    logger.info("Applying CLI overrides...")
    overrides_applied_count = 0
    for arg_key, config_path in cli_overrides_map.items():
        value = getattr(args, arg_key, None)
        # Special handling for verbose: store_true sets default=False,
        # we only override if the flag was explicitly passed OR if it wasn't passed but verbose is True in args
        if arg_key == 'verbose':
             if args.verbose: # If -v flag is present, override to True
                  if set_nested(effective_config, config_path, True):
                      overrides_applied_count += 1
             # No need for 'else', as the default is False which matches default config state
        elif value is not None: # Apply override if arg was provided
             if set_nested(effective_config, config_path, value):
                 overrides_applied_count += 1

    logger.info(f"Applied {overrides_applied_count} overrides from CLI arguments.")

    # Update log level again based on the final effective config's verbose setting
    final_verbose = effective_config.get('run_settings', {}).get('verbose', False)
    final_log_level = logging.DEBUG if final_verbose else logging.INFO
    logging.getLogger().setLevel(final_log_level)
    logger.info(f"Final log level set to: {logging.getLevelName(final_log_level)}")


    if args.show_config:
         import json
         print("--- Effective Configuration ---")
         print(json.dumps(effective_config, indent=4))
         print("-----------------------------")
         return # Exit after showing config

    # --- Get Final Run Settings ---
    run_settings = effective_config.get('run_settings', {})
    num_samples = run_settings.get('num_samples', 1) # Default to 1 if not set
    output_dir = run_settings.get('output_dir', './output_cli') # Default output dir
    master_seed = run_settings.get('master_seed', None)

    # --- Initialize Master RNG ---
    if master_seed is None:
        master_seed = random.randint(0, 2**32 - 1)
        logger.info(f"Generated master seed: {master_seed}")
    else:
         logger.info(f"Using provided master seed: {master_seed}")
    master_rng = get_rng(master_seed)
    # Store the seed actually used back into the effective config for record-keeping
    set_nested(effective_config, 'run_settings.master_seed_used', master_seed)


    # --- Ensure Output Directory Exists ---
    try:
        ensure_dir(output_dir)
        logger.info(f"Ensured output directory exists: {output_dir}")
    except Exception as e:
         logger.error(f"Failed to create output directory '{output_dir}': {e}", exc_info=True)
         return # Exit if output dir cannot be created

    # --- Generation Loop ---
    logger.info(f"Starting generation of {num_samples} samples to '{output_dir}'...")
    success_count = 0
    for i in range(num_samples):
        logger.info(f"--- Starting Sample {i} ---")
        try:
            # Pass the fully resolved effective_config
            if generate_sample(i, effective_config, master_rng, output_dir):
                success_count += 1
                logger.info(f"--- Successfully Completed Sample {i} ---")
            else:
                 logger.warning(f"--- Sample {i} generation function returned False ---")

        except Exception as e:
            logger.error(f"!!! CRITICAL ERROR generating sample {i} !!! : {e}", exc_info=True)
            # Option: stop or continue? Continue for now.

    logger.info(f"Generation finished. {success_count}/{num_samples} samples generated successfully.")

