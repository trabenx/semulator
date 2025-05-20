import sys
import argparse
import os
import random
import logging
import copy # Import copy for deepcopy
import json # Import json for logging
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

from src.core.configuration import load_config, set_nested, randomize_config_for_sample
from src.core.generator import generate_sample
from src.core.utils import get_rng, ensure_dir

# --- Wrapper Function for Multiprocessing ---
def generate_sample_wrapper(args_dict):
    """
    Wrapper to unpack arguments and call generate_sample.
    Designed to be called by ProcessPoolExecutor.map.
    """
    sample_idx = args_dict['sample_idx']
    sample_seed = args_dict['sample_seed']
    base_config = None
    try: # Randomize within the wrapper in the child process
         base_config = randomize_config_for_sample(copy.deepcopy(args_dict['base_config']), sample_seed)
    except Exception as e_rand_mp:
         # How to log this from child process effectively? Difficult.
         # Print might work, or need more complex logging setup.
         print(f"ERROR randomizing config in worker for sample {idx}: {e_rand_mp}")
         return False # Indicate failure
    if base_config is None:
        return False

    return generate_sample(sample_idx, sample_seed, base_config, args_dict['output_dir'])
# --- End Wrapper Function ---

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
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: CPU count)") # Add workers arg
    parser.add_argument("--start-index", type=int, default=0, help="Sample index to start generation from (0-based). Useful for resuming.")

    # Add other CLI overrides if needed, using dot notation for help text if desired
    # Example: parser.add_argument("--magnification", type=float, help="Override image_settings.magnification")
    return parser.parse_args() # Parses arguments from sys.argv (which main.py modified)

def run_cli():
    args = parse_cli_args()

    # --- Logging Setup ---
    # Base logging config might be set in main.py or here based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - [MainCLI] - %(message)s', force=True)
    # Configure root logger if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    else: # If already configured (e.g. by main.py), just set the level
         logging.getLogger().setLevel(log_level)

    logger = logging.getLogger(__name__)
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
    effective_config = copy.deepcopy(base_config)
    cli_overrides_map = {
        'output_dir': 'run_settings.output_dir',
        'num_samples': 'run_settings.num_samples',
        'seed': 'run_settings.master_seed',
        'verbose': 'run_settings.verbose'
    }
    overrides_applied_count = 0
    for arg_key, config_path in cli_overrides_map.items():
        value = getattr(args, arg_key, None)
        if value is not None or (arg_key == 'verbose' and args.verbose):
             if set_nested(effective_config, config_path, value if arg_key != 'verbose' else True):
                 overrides_applied_count += 1
    logger.info(f"Applied {overrides_applied_count} CLI overrides.")

    # --- Get Run Settings ---
    run_settings = effective_config.get('run_settings', {})
    num_samples = run_settings.get('num_samples', 1)
    output_dir = run_settings.get('output_dir', './output_cli')
    master_seed = run_settings.get('master_seed', None)

    # --- Get Start Index ---
    start_index = args.start_index
    if start_index < 0:
        logger.warning(f"Start index {start_index} is negative. Starting from 0.")
        start_index = 0
    if start_index >= num_samples:
        logger.info(f"Start index {start_index} is >= total number of samples {num_samples}. Nothing to generate.")
        return
    logger.info(f"Effective start index: {start_index}")
    # ---

    final_verbose = run_settings.get('verbose', False)
    final_log_level = logging.DEBUG if final_verbose else logging.INFO
    logging.getLogger().setLevel(final_log_level) # Set root logger level
    logger.info(f"Final log level set to: {logging.getLevelName(final_log_level)}")
    if args.show_config:
        import json;
        print(json.dumps(effective_config, indent=4));
        return

    # --- Initialize Master RNG & Generate Seeds ---
    if master_seed is None:
        master_seed = random.randint(0, 2**32 - 1)
        logger.info(f"Generated master seed: {master_seed}")
    else:
        logger.info(f"Using provided master seed: {master_seed}")
    master_rng = get_rng(master_seed)
    set_nested(effective_config, 'run_settings.master_seed_used', master_seed)

    logger.info(f"Generating {num_samples} sample seeds...")
    sample_seeds = [master_rng.randint(0, 2**32 - 1) for _ in range(num_samples)]
    logger.info("Sample seeds generated.")

    # --- Ensure Output Directory Exists ---
    try:
        ensure_dir(output_dir)
        logger.info(f"Ensured output directory exists: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory '{output_dir}': {e}", exc_info=True)
        return

    # --- Decide on Execution Mode (Sequential or Parallel) ---
    max_workers = args.workers if args.workers else os.cpu_count()
    use_multiprocessing = max_workers != 1 # Enable if --workers is set > 1
    success_count = 0
    if not use_multiprocessing:
        logger.info(f"Starting generation of {num_samples} samples (sequential mode [i.e., single worker])")
        for i in range(num_samples):
            sample_seed = sample_seeds[i]
            sample_config = None
            logger.info(f"--- [MainCLI] Starting Sample {i} (Seed: {sample_seed}) ---")

            # --- Add try-except AROUND randomization ---
            try:
                logger.debug(f"[MainCLI] Attempting config randomization for sample {i}")
                bg_types_in_base = effective_config.get('background', {}).get('types_choices', 'MISSING_KEY')
                logger.debug(f"[MainCLI] Base config background.types_choices: {bg_types_in_base}")

                # Pass the effective_config (base + CLI overrides)
                sample_config = randomize_config_for_sample(copy.deepcopy(effective_config), sample_seed)

                bg_type_in_sample = sample_config.get('background', {}).get('types', 'MISSING_KEY or list?')
                logger.debug(f"[MainCLI] Randomized config background.types: {bg_type_in_sample} (Type: {type(bg_type_in_sample)})")
                if bg_type_in_sample is None:
                     logger.error(f"[MainCLI] !!! Randomization resulted in None for background type for sample {i} !!!")

            except Exception as rand_err:
                 logger.error(f"[MainCLI] !!! EXCEPTION during config randomization for sample {i} !!! Error: {rand_err}", exc_info=True)
                 continue # Skip this sample
            # --- End try-except ---

            if sample_config is None:
                 logger.error(f"[MainCLI] sample_config is None after randomization block for sample {i}. Skipping.")
                 continue

            # --- Call generate_sample ---
            try:
                # Pass the now randomized sample_config as 'base_config' arg to generate_sample
                if generate_sample(i, sample_seed, sample_config, output_dir):
                    success_count += 1
                    logger.info(f"--- [MainCLI] Successfully Completed Sample {i} ---")
                else:
                     logger.warning(f"--- [MainCLI] Sample {i} generation returned failure (check sample log) ---")
            except Exception as e:
                 logger.error(f"!!! [MainCLI] CRITICAL UNCAUGHT ERROR during generate_sample for sample {i} !!! : {e}", exc_info=True)

    # --- Parallel Execution ---
    else:
        # --- Prepare Tasks for Multiprocessing ---
        tasks = []
        for i in range(num_samples):
            task_args = {
                'sample_idx': i,
                'sample_seed': sample_seeds[i],
                'base_config': effective_config, # Pass the fully resolved config
                'output_dir': output_dir,
            }
            tasks.append(task_args)

        # --- Execute in Parallel using ProcessPoolExecutor ---
        # Import necessary modules
        import concurrent.futures
        import tqdm # For progress bar (pip install tqdm)

        logger.info(f"Starting generation of {num_samples} samples using up to {max_workers} worker processes...")
        results = [] # Store boolean results (True/False) from generate_sample

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use executor.map which applies the function to each item in tasks
            # Wrap with tqdm for progress bar
            # chunksize can be adjusted for performance tuning, default is often fine
            results_iterator = executor.map(generate_sample_wrapper, tasks)
            # Process results as they complete
            for result in tqdm.tqdm(results_iterator, total=num_samples, desc="Generating Samples"):
                results.append(result)
                if result:
                    success_count += 1

        # Optionally report which samples failed if needed by checking 'results' list
        failed_indices = [i for i, success in enumerate(results) if not success]
        if failed_indices:
            logger.warning(f"Failed to generate samples with indices: {failed_indices}")
            logger.warning("Check the individual log files in the respective sample output directories for error details.")

    logger.info("Parallel generation finished.")
    logger.info(f"{success_count}/{num_samples} samples generated successfully.")