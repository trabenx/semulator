# tasks.py
import os
import time
import logging
import zipfile
import json
from pathlib import Path
from celery import shared_task, current_task
from celery.utils.log import get_task_logger
import copy # Import copy for deepcopy

# Assuming worker runs from project root where main.py is
try:
    # Need randomize_config_for_sample here now!
    from src.core.configuration import randomize_config_for_sample
    from src.core.generator import generate_sample
    from src.core.utils import get_rng, ensure_dir
except ImportError:
    import sys
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
         sys.path.insert(0, str(project_root))
    from src.core.configuration import randomize_config_for_sample
    from src.core.generator import generate_sample
    from src.core.utils import get_rng, ensure_dir

logger = get_task_logger(__name__)

# DB Import (keep as before)
try:
    from interfaces.webui.app import db, Task
except ImportError as e:
     logger.error(f"Could not import db/Task model: {e}. Status won't be updated in DB.")
     db = None
     Task = None



@shared_task(bind=True) # bind=True gives access to self (the task instance)
def run_generation_task(self, task_id, effective_config):
    """
    Celery task to generate a dataset based on provided configuration.
    Updates task status and progress via Celery backend and optionally DB.
    """
    task_output_dir = None # Define scope
    try:
        # --- Add Detailed Logging Here ---
        logger.info(f"[Task {task_id}] Received effective_config keys: {list(effective_config.keys())}")
        artifact_raffle_received = effective_config.get('artifact_raffle', 'MISSING')
        if isinstance(artifact_raffle_received, dict):
             shape_artifacts_received = artifact_raffle_received.get('categories',{}).get('shape', 'MISSING_SHAPE_CATEGORY')
             logger.info(f"[Task {task_id}] Received artifact_raffle['categories']['shape'] type: {type(shape_artifacts_received)}")
             # Log the content carefully, maybe just the names first
             if isinstance(shape_artifacts_received, list):
                  received_names = [item.get('name', 'NO_NAME') if isinstance(item, dict) else 'INVALID_ITEM_TYPE' for item in shape_artifacts_received]
                  logger.info(f"[Task {task_id}] Received shape artifact names: {received_names}")
                  # Optionally log the full structure if needed (can be verbose)
                  # logger.debug(f"[Task {task_id}] Full received shape artifacts: {json.dumps(shape_artifacts_received, indent=2)}")
             else:
                 logger.error(f"[Task {task_id}] Received shape artifacts section is not a list: {shape_artifacts_received}")

        else:
             logger.error(f"[Task {task_id}] Received artifact_raffle section is not a dict or is missing: {artifact_raffle_received}")
        # --- End Detailed Logging ---
        task_record = None
        if Task and db: # Check if DB import succeeded
            task_record = Task.query.get(task_id)
            if task_record:
                task_record.status = 'RUNNING'
                task_record.celery_task_id = self.request.id # Store celery's internal ID
                db.session.commit()
            else:
                 logger.warning(f"Task record ID {task_id} not found in DB.")
        else:
             logger.warning("DB or Task model not available. Cannot update DB status.")

        # Get settings from config passed to the task
        run_settings = effective_config.get('run_settings', {})
        num_samples = run_settings.get('num_samples', 1)
        # Use a dedicated output dir based on task ID
        # Ensure OUTPUT_DIR is configured correctly (e.g., via app context or env var)
        base_output_dir = effective_config.get('_output_base_dir', './output_web') # Pass this base dir if needed
        task_output_dir = Path(base_output_dir) / f"task_{task_id}"
        ensure_dir(task_output_dir)

        master_seed = run_settings.get('master_seed_used', run_settings.get('master_seed'))
        if master_seed is None:
            master_seed = int(time.time()) # Generate seed if somehow still missing
        master_rng = get_rng(master_seed)
        sample_seeds = [master_rng.randint(0, 2**32 - 1) for _ in range(num_samples)]

        logger.info(f"[Task {task_id}] Starting generation of {num_samples} samples to {task_output_dir} (Master Seed: {master_seed})")
        success_count = 0
        total_generated = 0

        for i in range(num_samples):
            total_generated = i + 1
            sample_seed = sample_seeds[i]
            logger.info(f"[Task {task_id}] Generating sample {i} with seed {sample_seed}")

            try:
                # Pass the ORIGINAL base_config received by the task
                # Make a deep copy before randomizing if base_config might be reused (safer)
                sample_config = randomize_config_for_sample(copy.deepcopy(base_config), sample_seed)
                logger.debug(f"[Task {task_id}, Sample {i}] Config randomized.")
            except Exception as rand_err:
                 logger.error(f"[Task {task_id}, Sample {i}] Failed to randomize config: {rand_err}", exc_info=True)
                 # Update progress/status to show error for this sample? Skip sample?
                 self.update_state(state='PROGRESS', meta={'current': i + 1, 'total': num_samples, 'status': f'Error randomizing config for sample {i+1}'})
                 continue # Skip this sample if config randomization fails


            # Update progress via Celery state meta
            progress = int(((i + 1) / num_samples) * 100)
            self.update_state(state='PROGRESS', meta={'current': i + 1, 'total': num_samples, 'status': f'Generating sample {i+1}/{num_samples}'})

            # Update DB progress (less frequent might be better)
            if Task and db and task_record and (i % 5 == 0 or i == num_samples - 1): # Update every 5 samples or on last
                 try:
                      task_record.progress = progress
                      db.session.commit()
                 except Exception as db_err:
                      logger.error(f"[Task {task_id}] Error updating DB progress: {db_err}")
                      db.session.rollback() # Rollback session on error


            # Call the actual sample generation function
            success = generate_sample(
                sample_idx=i,
                sample_seed=sample_seed,
                base_config=effective_config,
                output_parent_dir=task_output_dir
            )
            if success:
                success_count += 1
            else:
                 logger.warning(f"[Task {task_id}] Sample {i} generation failed (check sample log).")


        logger.info(f"[Task {task_id}] Generation loop finished. {success_count}/{num_samples} succeeded.")

        # --- Create ZIP file ---
        zip_filename = f"task_{task_id}_results.zip"
        zip_filepath = task_output_dir.parent / zip_filename # Store zip next to task dir
        logger.info(f"[Task {task_id}] Creating results ZIP file: {zip_filepath}")
        try:
            with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_STORED) as zf: # ZIP_STORED = no compression
                 for root, dirs, files in os.walk(task_output_dir):
                     for file in files:
                         file_path = Path(root) / file
                         arcname = file_path.relative_to(task_output_dir)
                         zf.write(file_path, arcname=arcname)
            logger.info(f"[Task {task_id}] ZIP file created successfully.")
            final_status = 'COMPLETED'
            final_message = f"Completed {success_count}/{num_samples} samples."
            result_info = {'status': final_status, 'result_path': str(zip_filepath), 'message': final_message}
        except Exception as zip_err:
            logger.error(f"[Task {task_id}] Failed to create ZIP file: {zip_err}", exc_info=True)
            final_status = 'FAILED'
            final_message = f"Generation finished ({success_count}/{num_samples}) but failed to create ZIP."
            result_info = {'status': final_status, 'error': str(zip_err), 'message': final_message}


        # --- Update final DB status ---
        if Task and db and task_record:
            try:
                 task_record.status = final_status
                 task_record.progress = 100
                 task_record.result_path = str(zip_filepath) if final_status == 'COMPLETED' else None
                 task_record.message = final_message
                 db.session.commit()
            except Exception as db_err:
                 logger.error(f"[Task {task_id}] Error updating final DB status: {db_err}")
                 db.session.rollback()

        # Return result info for Celery backend
        return result_info

    except Exception as e:
        logger.error(f"[Task {task_id}] UNHANDLED EXCEPTION in generation task: {e}", exc_info=True)
        # Update Celery state
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        # Update DB status
        if Task and db and task_record:
            try:
                 task_record.status = 'FAILED'
                 task_record.message = f"Task failed: {e}"
                 db.session.commit()
            except Exception as db_err:
                 logger.error(f"[Task {task_id}] Error updating FAILED DB status: {db_err}")
                 db.session.rollback()
        # Re-raise the exception so Celery marks the task as failed
        raise e
