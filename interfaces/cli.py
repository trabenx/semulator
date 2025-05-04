import argparse
import os
import random
import logging
import imageio
import copy
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

# --- Wrapper Function for Multiprocessing ---
def generate_sample_wrapper(args_dict):
    """
    Wrapper to unpack arguments and call generate_sample.
    Designed to be called by ProcessPoolExecutor.map.
    """
    sample_idx = args_dict['sample_idx']
    sample_seed = args_dict['sample_seed']
    base_config = args_dict['base_config']
    output_dir = args_dict['output_dir']
    # Optional: Configure logging here for the child process if needed,
    # but generate_sample now handles its own file logging.

    # Call the actual generation function
    # Return value indicates success/failure
    return generate_sample(sample_idx, sample_seed, base_config, output_dir)
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
    # Use a basic config for the main process messages
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - [Main] - %(message)s', force=True)
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
    effective_config = copy.deepcopy(base_config) # Use deepcopy
    cli_overrides_map = {
        'output_dir': 'run_settings.output_dir', 
        'num_samples': 'run_settings.num_samples', 
        'seed': 'run_settings.master_seed', 
        'verbose': 'run_settings.verbose'
    }
    
    for arg_key, config_path in cli_overrides_map.items():
        value = getattr(args, arg_key, None)
        if value is not None or (arg_key == 'verbose' and args.verbose):
            set_nested(effective_config, config_path, value if arg_key != 'verbose' else True)
    run_settings = effective_config.get('run_settings', {})
    num_samples = run_settings.get('num_samples', 1)
    output_dir = run_settings.get('output_dir', './output_cli')
    master_seed = run_settings.get('master_seed', None)
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
    try: ensure_dir(output_dir); logger.info(f"Ensured output directory exists: {output_dir}")
    except Exception as e: logger.error(f"Failed to create output directory '{output_dir}': {e}", exc_info=True); return


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
    import os # For cpu_count

    max_workers = args.workers if args.workers else os.cpu_count()
    logger.info(f"Starting generation of {num_samples} samples using up to {max_workers} worker processes...")
    success_count = 0
    results = [] # Store boolean results (True/False) from generate_sample

    # Use ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use executor.map which applies the function to each item in tasks
        # Wrap with tqdm for progress bar
        # chunksize can be adjusted for performance tuning, default is often fine
        results_iterator = executor.map(generate_sample_wrapper, tasks)
        # Process results as they complete
        for result in tqdm.tqdm(results_iterator, total=num_samples, desc="Generating Samples"):
            results.append(result)
            if result: # Check if generate_sample returned True
                success_count += 1

    logger.info(f"Parallel generation finished.")
    logger.info(f"{success_count}/{num_samples} samples generated successfully.")

    # Optionally report which samples failed if needed by checking 'results' list
    failed_indices = [i for i, success in enumerate(results) if not success]
    if failed_indices:
        logger.warning(f"Failed to generate samples with indices: {failed_indices}")
        logger.warning("Check the individual log files in the respective sample output directories for error details.")

