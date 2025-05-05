import sys
import os
import logging
import json
import random
import threading
import time
import copy
from io import BytesIO
import zipfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, url_for, redirect
from flask_sqlalchemy import SQLAlchemy
import datetime
import collections.abc # For deep merge


# --- Add src path ---
project_root = Path(__file__).resolve().parent.parent.parent # Up 3 levels
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# --- Import Core Components ---
from src.core.configuration import load_config, override_config, set_nested, randomize_config_for_sample
from src.core.generator import generate_sample
from src.core.utils import get_rng, ensure_dir, image_to_bit_depth

from celery_app import celery

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', f"sqlite:///{project_root / 'dev.db'}")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
app.config['BASE_CONFIG_PATH'] = str(project_root / 'config' / 'base_config.json')
app.config['WEB_OUTPUT_DIR'] = os.environ.get('WEB_OUTPUT_DIR', str(project_root / 'output_web'))
app.config['UPLOAD_FOLDER'] = str(project_root / 'uploads')
ensure_dir(app.config['WEB_OUTPUT_DIR'])
ensure_dir(app.config['UPLOAD_FOLDER'])

# --- Import Models AFTER db is initialized ---
from .models import Task, LayerDefinition

# --- Basic Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = app.logger
logger.setLevel(logging.INFO)

# --- In-memory job tracking (Replace with DB or Redis for production) ---
jobs = {} # { job_id: {'status': 'running/completed/failed', 'progress': 0, 'output_dir': ...} }
job_counter = 0
job_lock = threading.Lock()


# --- Generation Worker Function ---
def generation_worker(job_id, effective_config):
    global jobs
    run_settings = effective_config.get('run_settings', {})
    num_samples = run_settings.get('num_samples', 1)
    job_output_dir = Path(app.config['OUTPUT_DIR']) / f"job_{job_id}"
    master_seed = run_settings.get('master_seed', None)

    ensure_dir(job_output_dir)
    if master_seed is None:
        master_seed = random.randint(0, 2**32 - 1)
        logger.info(f"[Job {job_id}] Generated master seed: {master_seed}")
    else:
        logger.info(f"[Job {job_id}] Using provided master seed: {master_seed}")

    master_rng = get_rng(master_seed)

    logger.info(f"[Job {job_id}] Generating {num_samples} sample seeds...")
    sample_seeds = [master_rng.randint(0, 2**32 - 1) for _ in range(num_samples)]
    logger.info(f"[Job {job_id}] Sample seeds generated.")

    logger.info(f"Job {job_id}: Starting generation of {num_samples} samples (Seed: {master_seed})")
    success_count = 0
    try:
        for i in range(num_samples):
            progress = int(((i + 1) / num_samples) * 100)
            sample_seed_for_worker = sample_seeds[i] # Get the specific seed for this sample
            with job_lock:
                jobs[job_id]['status'] = 'running'
                jobs[job_id]['progress'] = progress
                jobs[job_id]['message'] = f"Generating sample {i+1}/{num_samples} (Seed: {sample_seed_for_worker})"

            logger.info(f"Job {job_id}: Generating sample {i} with seed {sample_seed_for_worker}")
            success = generate_sample(
                sample_idx=i,
                sample_seed=sample_seed_for_worker, # Pass integer seed
                base_config=effective_config,      # Pass full config
                output_parent_dir=job_output_dir   # Pass job output dir
            )
            if success:
                success_count += 1
            else:
                 logger.warning(f"[Job {job_id}] Sample {i} generation returned failure.")
            # time.sleep(0.1) # Remove sleep unless debugging rate limiting issues

        with job_lock:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['progress'] = 100
            jobs[job_id]['message'] = f"Completed {success_count}/{num_samples} samples."
            jobs[job_id]['output_dir'] = str(job_output_dir)
        logger.info(f"Job {job_id}: Completed successfully ({success_count}/{num_samples}).")

    except Exception as e:
        logger.error(f"Job {job_id}: Failed during generation - {e}", exc_info=True)
        with job_lock:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = str(e)
            jobs[job_id]['message'] = f"Error during generation: {e}"


def deep_update(source, overrides):
    """
    Recursively update a dict-like structure (source) with values from another (overrides).
    Modifies 'source' potentially if mutable, but safer to use return value.
    Handles list vs dict type mismatches.
    Merges lists by index: applies override item N to source item N.
    """
    # If source is not a dictionary, simply return a deep copy of the override
    # unless the override is None (which might mean no override was intended)
    if not isinstance(source, collections.abc.Mapping):
        return copy.deepcopy(overrides) if overrides is not None else source

    # Work on a copy of the source dictionary
    output = copy.deepcopy(source)

    for key, override_value in overrides.items():
        source_value = output.get(key) # Get current value from the copied source

        # --- Case 1: Override value is a dictionary ---
        if isinstance(override_value, collections.abc.Mapping) and override_value:
            if isinstance(source_value, collections.abc.Mapping):
                # Both are dicts, recurse
                output[key] = deep_update(source_value, override_value)
            else:
                # Source value is not a dict (or None), replace with override dict
                output[key] = copy.deepcopy(override_value)

        # --- Case 2: Override value is a list ---
        elif isinstance(override_value, list):
            # Check if source value is also a list
            if isinstance(source_value, list):
                # --- Merge lists element by element (by index) ---
                merged_list = []
                len_source = len(source_value)
                len_override = len(override_value)
                max_len = max(len_source, len_override)

                for i in range(max_len):
                    src_item = source_value[i] if i < len_source else None
                    ovr_item = override_value[i] if i < len_override else None

                    if ovr_item is None:
                        # No override for this index, keep source item
                        merged_list.append(copy.deepcopy(src_item) if src_item is not None else None)
                    elif src_item is None:
                        # No source item for this index, add override item
                        merged_list.append(copy.deepcopy(ovr_item))
                    elif isinstance(src_item, collections.abc.Mapping) and isinstance(ovr_item, collections.abc.Mapping):
                        # Both are dicts, recursively merge them
                        merged_list.append(deep_update(src_item, ovr_item))
                    else:
                        # Structures mismatch or not dicts, override takes precedence
                        merged_list.append(copy.deepcopy(ovr_item))
                output[key] = merged_list
                # logger.debug(f"Index-merged list for key '{key}'")
            else:
                # Source is not a list, override replaces it entirely
                output[key] = copy.deepcopy(override_value)

        # --- Case 3: Override value is primitive (or None) ---
        else:
            # Directly set/overwrite the value in the output dict
            output[key] = override_value

    return output



def parse_form_to_overrides(form_data):
    """
    Parses Flask form data (ImmutableMultiDict) into a nested dictionary of overrides.
    Assumes form field names use '.' for nesting (e.g., 'a.b.c').
    Handles basic type conversions (int, float, bool).
    Handles '.min'/'.max' suffixes for ranges.
    Handles keys corresponding to checkboxes (assumes key presence means 'True').
    """
    overrides = {}
    range_partials = {} # To collect min/max pairs

    # Use the Python set_nested helper defined earlier in this file or imported
    # def set_nested(d, keys, value, create_missing=True): ...

    processed_checkbox_keys = set() # Track checkboxes handled by presence

    # Checkbox handling: Iterate keys first to identify potential checkboxes
    # A checkbox is only present in form_data if it was checked ('on')
    # We need to infer 'False' for checkboxes that are *not* in the form_data
    # but this requires knowing the full expected structure (difficult here).
    # Simpler approach: If a key exists and corresponds to a known boolean field
    # treat its presence as True. Otherwise, assume False if not present? Risky.
    # Safest approach: Rely on the JS sending JSON where unchecked boxes are explicitly false.
    # For now, let's parse what's *in* the form: presence implies True for relevant keys.

    for key, value_str in form_data.items(multi=True): # Use multi=True if keys can repeat? Usually not for this form.
        # Skip empty values unless it's meant to clear something (tricky)
        if not value_str:
             continue

        parsed_value = None

        # Handle range suffixes (.min, .max)
        is_range_min = key.endswith('.min')
        is_range_max = key.endswith('.max')

        if is_range_min or is_range_max:
            base_key = key.rsplit('.', 1)[0]
            suffix = key.rsplit('.', 1)[1] # 'min' or 'max'
            if base_key not in range_partials: range_partials[base_key] = {}
            try: # Convert range values to numbers
                val = float(value_str)
                if val.is_integer(): val = int(val)
                range_partials[base_key][suffix] = val
            except ValueError:
                 logger.warning(f"Could not parse range value for {key}: '{value_str}'")
            continue # Move to next form item

        # Handle regular keys - Attempt conversions
        try:
             parsed_value = int(value_str)
        except ValueError:
             try:
                  parsed_value = float(value_str)
             except ValueError:
                  # Explicit check for boolean strings
                  if value_str.lower() in ['true', 'on', 'yes']:
                       parsed_value = True
                       # Mark if it was a checkbox based on value='on' convention
                       if value_str.lower() == 'on': processed_checkbox_keys.add(key)
                  elif value_str.lower() in ['false', 'off', 'no']:
                       parsed_value = False
                  else:
                       parsed_value = value_str # Keep as string

        # Set the value in the nested overrides dictionary
        # Use the Python set_nested function
        set_nested(overrides, key, parsed_value)

    # Process collected range partials
    for base_key, parts in range_partials.items():
        if 'min' in parts and 'max' in parts:
             # Add _range suffix back for consistency with JSON/Python config
             set_nested(overrides, base_key + '_range', [parts['min'], parts['max']])
        else:
             logger.warning(f"Incomplete range found for {base_key}: {parts}")

    # Post-process checkbox keys - if a key was identified as checkbox ('on'), ensure it's True
    # This step might be redundant if 'on' was already parsed to True above.
    # A more robust way would be needed to handle *unchecked* boxes if not sending JSON.
    # for key in processed_checkbox_keys:
    #      set_nested(overrides, key, True) # Ensure it's True

    logger.debug(f"Parsed form overrides (Python): {json.dumps(overrides, indent=2)}")
    return overrides


# --- Routes ---
@app.route('/')
def index():
    """ Renders the main generation configuration page. """
    try:
        base_config = load_config(app.config['BASE_CONFIG_PATH'])
        # Pass config as JSON string for JS to parse and build form
        return render_template('generate.html', config_json=json.dumps(base_config))
    except Exception as e:
         logger.error(f"Failed to load base config for UI: {e}", exc_info=True)
         return f"Error loading configuration: {e}", 500


@app.route('/generate', methods=['POST'])
def start_generation():
    global job_counter, jobs
    logger.info("Received generation request.")

    # --- Load base config ---
    try:
        # Load a fresh copy each time to avoid modifying the base in memory
        base_config = load_config(app.config['BASE_CONFIG_PATH'])
        logger.debug("Base config loaded.")
    except Exception as e:
         logger.error(f"Failed to load base config: {e}")
         return jsonify({'status': 'error', 'message': f'Failed to load base config: {e}'}), 500

    # --- Parse form data into structured overrides ---
    try:
        form_overrides = parse_form_to_overrides(request.form)
        logger.info("Form overrides parsed.")
    except Exception as e:
        logger.error(f"Failed to parse form data: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'Failed to parse form data: {e}'}), 400

    # --- Create effective config by merging base and overrides ---
    effective_config = {} # Start with empty dict
    try:
        # Deep copy base config first to avoid modifying the original loaded dict
        effective_config = json.loads(json.dumps(base_config)) # Simple deep copy via JSON
        # Recursively update the copy with the parsed overrides
        effective_config = deep_update(effective_config, form_overrides)
        logger.debug("Effective config created after merging overrides.")

        # Ensure essential run_settings exist if overridden partially
        if 'run_settings' not in effective_config: effective_config['run_settings'] = {}
        effective_config['run_settings'].setdefault('num_samples', 1) # Ensure default if not set
        effective_config['run_settings'].setdefault('master_seed', None)
        effective_config['run_settings'].setdefault('verbose', True) # Default verbose for web logs

    except Exception as e:
        logger.error(f"Failed to merge overrides into config: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'Failed to merge configuration: {e}'}), 500


    # --- Start background job ---
    with job_lock:
         job_id = job_counter
         job_counter += 1
         jobs[job_id] = {'status': 'queued', 'progress': 0, 'message': 'Job queued'}
         logger.info(f"Job {job_id} created and queued.")

    # Pass the fully resolved effective_config to the worker
    thread = threading.Thread(target=generation_worker, args=(job_id, effective_config))
    thread.daemon = True # Allow app to exit even if worker threads are running (optional)
    thread.start()
    logger.info(f"Generation thread started for Job {job_id}.")

    return jsonify({'status': 'started', 'job_id': job_id})

@app.route('/status/<int:job_id>')
def get_status(job_id):
    with job_lock:
        job = jobs.get(job_id)
    if job:
        return jsonify(job)
    else:
        return jsonify({'status': 'error', 'message': 'Job not found'}), 404

@app.route('/results/<int:job_id>')
def list_results(job_id):
     with job_lock: job = jobs.get(job_id)
     if not job or job['status'] != 'completed':
         return jsonify({'status': 'error', 'message': 'Job not found or not completed'}), 404

     job_dir = Path(job['output_dir'])
     files = []
     try:
         # List primary images
         for item in sorted(job_dir.glob('sem_*.*')):
             if item.is_file() and item.suffix in ['.tif', '.tiff', '.png']:
                  files.append({'name': item.name, 'url': f'/download/{job_id}/{item.name}'})
         # List sample subdirectories
         for sample_dir in sorted(job_dir.glob('sem_*/')):
             if sample_dir.is_dir():
                  files.append({'name': f"{sample_dir.name}/ (view details)", 'url': f'/results/{job_id}/{sample_dir.name}'})

         # Add link to download zip
         files.append({'name': 'Download All (ZIP)', 'url': f'/download_zip/{job_id}'})
     except Exception as e:
         logger.error(f"Error listing results for job {job_id}: {e}")
         return jsonify({'status': 'error', 'message': 'Error listing results'}), 500
     return jsonify({'job_id': job_id, 'status': 'completed', 'files': files, 'message': job.get('message','')})

# Route to view files within a sample subdirectory
@app.route('/results/<int:job_id>/<sample_name>')
def list_sample_details(job_id, sample_name):
    with job_lock: job = jobs.get(job_id)
    if not job or 'output_dir' not in job: return "Job not found", 404

    job_dir = Path(job['output_dir'])
    sample_dir = job_dir / sample_name
    if not sample_dir.is_dir() or sample_dir.parent != job_dir: # Basic path check
         return "Sample directory not found", 404

    files = []
    try:
        # List files directly in the sample dir
        for item in sorted(sample_dir.glob('*.*')):
             if item.is_file():
                  files.append({'name': item.name, 'url': f'/download/{job_id}/{sample_name}/{item.name}'})
        # List layer subdirs
        for layer_dir in sorted(sample_dir.glob('layers/layer_*/')):
             if layer_dir.is_dir():
                 files.append({'name': f"{layer_dir.relative_to(sample_dir)}/ (view layers)", 'url': f'/results/{job_id}/{sample_name}/{layer_dir.relative_to(sample_dir)}'})

        # Simplified - could recursively list all files/dirs
    except Exception as e:
        logger.error(f"Error listing sample details for {sample_name}, job {job_id}: {e}")
        return jsonify({'status': 'error', 'message': 'Error listing sample files'}), 500
    return jsonify({'job_id': job_id, 'sample_name': sample_name, 'files': files})

@app.route('/download/<int:job_id>/<path:filename>')
def download_file(job_id, filename):
    with job_lock:
        job = jobs.get(job_id)

    if not job or 'output_dir' not in job:
        return "Job not found or invalid", 404

    # Security: Ensure filename is safe (e.g., doesn't contain '..')
    safe_filename = Path(filename).name # Basic sanitization
    if safe_filename != filename:
         return "Invalid filename", 400

    directory = Path(job['output_dir'])
    try:
        # Ensure the file requested is directly within the job's output dir
        # or one level down (e.g., in sem_xxxxx/ ) - adjust glob pattern if needed
        allowed_paths = list(directory.glob(f"{safe_filename}")) + \
                        list(directory.glob(f"sem_*/{safe_filename}")) + \
                        list(directory.glob(f"sem_*/layers/*/{safe_filename}")) # Add more levels if needed

        if not allowed_paths or not allowed_paths[0].is_file():
             logger.warning(f"Attempt to download non-existent or disallowed file: {filename} from job {job_id}")
             return "File not found", 404

        # Use the first match's directory for send_from_directory
        file_to_send_dir = allowed_paths[0].parent

        return send_from_directory(directory=file_to_send_dir, path=safe_filename, as_attachment=True)
    except Exception as e:
        logger.error(f"Error serving file {filename} for job {job_id}: {e}")
        return "Error downloading file", 500

# Route to download files from subdirs (needs adjustment in download_file)
@app.route('/download/<int:job_id>/<path:filepath>')
def download_file_deep(job_id, filepath):
    with job_lock: job = jobs.get(job_id)
    if not job or 'output_dir' not in job: return "Job not found or invalid", 404

    job_dir = Path(job['output_dir'])
    target_file = job_dir / filepath

    # Security: Crucial check - ensure the resolved path is still within the job directory!
    if not target_file.is_file() or not target_file.resolve().is_relative_to(job_dir.resolve()):
        logger.warning(f"Attempt to download non-existent or disallowed file: {filepath} from job {job_id}")
        return "File not found or access denied", 404

    try:
         return send_from_directory(directory=target_file.parent, path=target_file.name, as_attachment=True)
    except Exception as e:
         logger.error(f"Error serving file {filepath} for job {job_id}: {e}")
         return "Error downloading file", 500


# Route to download ZIP
@app.route('/download_zip/<int:job_id>')
def download_zip(job_id):
    with job_lock: job = jobs.get(job_id)
    if not job or job['status'] != 'completed':
        return "Job not found or not completed", 404

    job_dir = Path(job['output_dir'])
    zip_filename = f"job_{job_id}_results.zip"
    memory_file = BytesIO()

    try:
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
             for root, dirs, files in os.walk(job_dir):
                 for file in files:
                     file_path = Path(root) / file
                     # Arcname is the path inside the zip file (relative to job_dir)
                     arcname = file_path.relative_to(job_dir)
                     zf.write(file_path, arcname=arcname)

        memory_file.seek(0)
        return send_file(memory_file, download_name=zip_filename, as_attachment=True, mimetype='application/zip')

    except Exception as e:
         logger.error(f"Error creating ZIP for job {job_id}: {e}")
         return "Error creating ZIP file", 500


@app.route('/preview', methods=['POST'])
def generate_preview():
    """ Generates a single sample synchronously for preview. """
    logger.info("Received preview request.")
    try:
        if not request.is_json:
            return jsonify({'status': 'error', 'message': 'Request must be JSON.'}), 400

        payload = request.get_json()
        received_base_config = payload.get('base_config')
        form_overrides = payload.get('overrides')

        if not isinstance(received_base_config, dict) or not isinstance(form_overrides, dict):
            return jsonify({'status': 'error', 'message': 'Invalid payload structure.'}), 400

        # --- Merge overrides onto the received base config ---
        # Use deepcopy to avoid modifying the received base if it's reused
        effective_config = deep_update(copy.deepcopy(received_base_config), form_overrides)
        logger.debug("Preview effective config created after merging overrides.")
        # ---


        # Ensure necessary run settings are present
        if 'run_settings' not in effective_config:
            effective_config['run_settings'] = {}
        preview_seed = effective_config['run_settings'].get('master_seed') # Use seed from effective config
        if not isinstance(preview_seed, int):
            preview_seed = random.randint(0, 2**32-1)
        effective_config['run_settings']['master_seed_used'] = preview_seed
        temp_output_dir = Path(app.config['WEB_OUTPUT_DIR']) / f"preview_{preview_seed}"
        ensure_dir(temp_output_dir)
        sample_seed = get_rng(preview_seed).randint(0, 2**32 - 1)

        # Pass the fully merged effective_config to generate_sample,
        # which will then call randomize_config_for_sample internally
        success = generate_sample(0, sample_seed, effective_config, temp_output_dir)
        # ---
        logger.info(f"Preview sample generation finished. Success: {success}")
        if success:
            preview_img_path_png = temp_output_dir / "sem_00000" / "image_final_noisy_vis.png"
            preview_img_path_tif = temp_output_dir / "sem_00000.tif"
            preview_img_path = preview_img_path_png if preview_img_path_png.is_file() else preview_img_path_tif
            logger.info(f"Looking for preview image at: {preview_img_path}")

            if preview_img_path.is_file():
                import base64
                import imageio
                logger.info("Preview image found. Reading bytes...")
                img_bytes = preview_img_path.read_bytes()
                mime_type = f'image/{preview_img_path.suffix.lower().strip(".")}'
                logger.info(f"Read {len(img_bytes)} bytes, mime_type: {mime_type}")
                if mime_type == 'image/tif' or mime_type == 'image/tiff':
                    logger.info("Attempting TIFF to PNG conversion...")
                    try: # Attempt conversion to PNG for browser
                        img_arr = imageio.v3.imread(preview_img_path)
                        logger.debug(f"Read TIFF array shape: {img_arr.shape}, dtype: {img_arr.dtype}")
                        if img_arr.dtype == np.uint16:
                            img_arr_u8 = (img_arr / 256).astype(np.uint8)
                        elif img_arr.dtype != np.uint8: # If not uint16 or uint8, attempt basic scale
                            img_arr_u8 = np.clip(img_arr, 0, 255).astype(np.uint8)
                        else:
                            img_arr_u8 = img_arr # Already uint8

                        png_stream = BytesIO()
                        imageio.imwrite(png_stream, img_arr_u8, format='png')
                        img_bytes = png_stream.getvalue()
                        mime_type = 'image/png'
                        logger.info("TIFF to PNG conversion successful.")
                    except Exception as conv_err:
                        logger.warning(f"Could not convert TIFF preview to PNG: {conv_err}")
                logger.info("Encoding image to Base64...")
                encoded_img = base64.b64encode(img_bytes).decode('utf-8')
                logger.info("Encoding successful. Returning preview.")
                return jsonify({'status': 'success', 'image_data': encoded_img, 'mime_type': mime_type})
            else:
                logger.error(f"Preview image file not found at expected path: {preview_img_path}")
                return jsonify({'status': 'error', 'message': 'Preview generated but output image not found.'}), 500
        else:
            logger.error("Preview generation function returned failure.")
            return jsonify({'status': 'error', 'message': 'Preview generation failed (check logs).'}), 500
    except Exception as e:
        logger.error(f"Error during preview generation: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'Preview error: {e}'}), 500

@app.route('/start_task', methods=['POST'])
def start_task():
    """ Starts a background generation task via Celery. """
    logger.info("Received request to start generation task.")
    try:
        if not request.is_json:
             return jsonify({'status': 'error', 'message': 'Invalid request format. Expected JSON.'}), 400

        payload = request.get_json()
        received_base_config = payload.get('base_config')
        form_overrides = payload.get('overrides')

        if not isinstance(received_base_config, dict) or not isinstance(form_overrides, dict):
            return jsonify({'status': 'error', 'message': 'Invalid payload structure.'}), 400

        # --- Merge overrides onto the received base config ---
        effective_config = deep_update(copy.deepcopy(received_base_config), form_overrides)
        # ---

        # Add output base dir for task worker reference
        effective_config['_output_base_dir'] = app.config['WEB_OUTPUT_DIR']
        # Ensure run settings and seed
        if 'run_settings' not in effective_config: effective_config['run_settings'] = {}
        effective_config['run_settings'].setdefault('num_samples', 1)
        if not isinstance(effective_config['run_settings'].get('master_seed'), int):
             effective_config['run_settings']['master_seed'] = random.randint(0, 2**32-1)
        effective_config['run_settings']['master_seed_used'] = effective_config['run_settings']['master_seed']

        # --- Create DB Task Record ---
        new_task = Task(
            status='PENDING',
            # Store the *effective* config that includes overrides
            config_json=json.dumps(effective_config)
        )
        db.session.add(new_task); db.session.commit(); task_db_id = new_task.id
        logger.info(f"Created DB task record with ID: {task_db_id}")

        # --- Launch Celery Task ---
        from tasks import run_generation_task
        # Pass the final effective_config to the task
        celery_task = run_generation_task.delay(task_db_id, effective_config)
        logger.info(f"Dispatched Celery task {celery_task.id} for DB task {task_db_id}")
        new_task.celery_task_id = celery_task.id; db.session.commit()

        return jsonify({'status': 'success', 'task_id': task_db_id, 'celery_id': celery_task.id})

    except Exception as e:
        logger.error(f"Error starting generation task: {e}", exc_info=True)
        db.session.rollback()
        return jsonify({'status': 'error', 'message': f'Error starting task: {e}'}), 500

@app.route('/tasks')
def view_tasks():
    """ Displays the list of running and completed tasks. """
    try:
        # Query tasks, order by start time descending
        tasks = Task.query.order_by(Task.start_time.desc()).limit(50).all() # Limit for pagination later
        return render_template('tasks.html', tasks=tasks)
    except Exception as e:
        logger.error(f"Error fetching tasks: {e}", exc_info=True)
        return "Error loading tasks page.", 500

@app.route('/task_status/<int:task_id>')
def get_task_status(task_id):
    """ Returns JSON status for AJAX polling. """
    try:
        task = Task.query.get_or_404(task_id)
        response = {
            'id': task.id,
            'status': task.status,
            'progress': task.progress,
            'message': task.message,
            'result_path': task.result_path,
            'celery_id': task.celery_task_id
        }
        # Optional: Get more detailed status from Celery backend if needed
        # if task.celery_task_id:
        #    async_result = celery.AsyncResult(task.celery_task_id)
        #    response['celery_state'] = async_result.state
        #    if async_result.state == 'PROGRESS': response['celery_meta'] = async_result.info
        #    elif async_result.state == 'FAILURE': response['celery_meta'] = str(async_result.info)

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error fetching status for task {task_id}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'Error fetching status: {e}'}), 500

@app.route('/download_task/<int:task_id>')
def download_task_result(task_id):
    """ Downloads the result ZIP file for a completed task. """
    try:
        task = Task.query.get_or_404(task_id)
        if task.status == 'COMPLETED' and task.result_path and Path(task.result_path).is_file():
            logger.info(f"Serving download for task {task_id}: {task.result_path}")
            # Ensure the path is absolute or relative to a known root
            file_path = Path(task.result_path)
            if not file_path.is_absolute():
                # If relative, assume it's relative to the project root or output dir
                # This depends on how the path was stored by the worker
                # Let's assume it stored an absolute path or one we can resolve easily
                pass # Assuming path stored is directly usable by send_file

            return send_file(file_path, as_attachment=True, download_name=file_path.name)
        elif task.status != 'COMPLETED':
             return "Task not completed.", 404
        else:
             logger.error(f"Result file not found for completed task {task_id}: {task.result_path}")
             return "Result file not found.", 404
    except Exception as e:
        logger.error(f"Error downloading result for task {task_id}: {e}", exc_info=True)
        return "Error processing download.", 500

# --- Config Import/Export Routes ---
# @app.route('/export_config', methods=['POST']) ...
# @app.route('/import_config', methods=['POST']) ...

# --- Layer Configuration Routes (Placeholder) ---
@app.route('/layers')
def view_layers():
     # TODO: Query LayerDefinition model and render template
     return "Layer configuration page (Not Implemented Yet)", 501

# @app.route('/add_layer', methods=['POST']) ...
# @app.route('/delete_layer/<int:layer_def_id>', methods=['POST']) ...


# --- Initialize DB command ---
@app.cli.command('init-db')
def init_db_command():
    """Creates the database tables."""
    logger.info("Initializing database...")
    try:
        db.create_all()
        logger.info('Database initialized.')
    except Exception as e:
         logger.error(f"Error initializing database: {e}", exc_info=True)


# --- Run ---
def run_webui():
    # Make sure DB exists before running
    db_path = Path(app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', ''))
    if not db_path.exists():
         logger.warning(f"Database file not found at {db_path}. Creating tables.")
         logger.warning("Run 'flask init-db' from the command line in your project root if needed.")
         with app.app_context(): # Need app context for db.create_all()
             try:
                 db.create_all()
                 logger.info("Database tables created.")
             except Exception as e:
                  logger.error(f"Failed to create database tables automatically: {e}")

    logger.info(f"Starting Flask Web UI. Output base: {app.config['WEB_OUTPUT_DIR']}")
    # Use waitress or gunicorn in production
    app.run(debug=False, host='0.0.0.0', port=5000)



# Example of running webui directly (can be called from main.py)
# if __name__ == "__main__":
#     run_webui()
