import sys
from pathlib import Path
import os
import logging
import collections.abc # For deep merge
import json
import random
import threading
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
import zipfile
from io import BytesIO

# --- Add src path ---
project_root = Path(__file__).resolve().parent.parent.parent # Up 3 levels
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# --- Import Core Components ---
from src.core.configuration import load_config, override_config
from src.core.generator import generate_sample
from src.core.utils import get_rng, ensure_dir

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24) # For session management if needed later
app.config['BASE_CONFIG_PATH'] = str(project_root / 'config' / 'base_config.json')
app.config['OUTPUT_DIR'] = str(project_root / 'output_web') # Separate output for web

# --- Basic Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger('FlaskWebUI')

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
    if master_seed is None: master_seed = random.randint(0, 2**32 - 1)
    master_rng = get_rng(master_seed)

    logger.info(f"Job {job_id}: Starting generation of {num_samples} samples (Seed: {master_seed})")
    success_count = 0
    try:
        for i in range(num_samples):
            progress = int(((i + 1) / num_samples) * 100)
            with job_lock:
                jobs[job_id]['status'] = 'running'
                jobs[job_id]['progress'] = progress
                jobs[job_id]['message'] = f"Generating sample {i+1}/{num_samples}"

            logger.info(f"Job {job_id}: Generating sample {i}/{num_samples-1}")
            if generate_sample(i, effective_config, master_rng, job_output_dir):
                success_count += 1
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


# --- Routes ---
@app.route('/')
def index():
    # Load base config to populate the form defaults (simplified)
    try:
        base_config = load_config(app.config['BASE_CONFIG_PATH'])
        # Pass relevant parts of config to template
        # This needs significant JS on the frontend to be truly dynamic
        return render_template('index.html', config=base_config)
    except Exception as e:
         logger.error(f"Failed to load base config for UI: {e}")
         return f"Error loading configuration: {e}", 500

def deep_update(source, overrides):
    """
    Recursively update a dict with values from another dict.
    Modifies 'source' in place.
    """
    for key, value in overrides.items():
        if isinstance(value, collections.abc.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        elif isinstance(value, list) and value:
            # Decide on list update strategy: replace or extend? Replace for simplicity now.
            source[key] = value
        else:
            source[key] = overrides[key]
    return source


def parse_form_to_overrides(form_data):
    """
    Parses Flask form data into a nested dictionary of overrides.
    Assumes form field names use '.' for nesting (e.g., 'a.b.c').
    Handles basic type conversions (int, float, bool).
    Handles '.min'/'.max' suffixes for ranges.
    Handles '.enabled' suffix for boolean toggles (value 'on' means True).
    """
    overrides = {}
    range_partials = {} # To collect min/max pairs

    # Helper to set nested value
    def set_nested_value(d, keys, value):
        keys_list = keys.split('.')
        current_level = d
        for key in keys_list[:-1]:
            if key not in current_level or not isinstance(current_level[key], dict):
                current_level[key] = {} # Create/overwrite intermediate dicts
            current_level = current_level[key]
        current_level[keys_list[-1]] = value

    for key, value_str in form_data.items():
        # Skip empty values unless it's a known boolean toggle
        if not value_str and not key.endswith('.enabled'):
            continue

        # Handle range suffixes (.min, .max)
        is_range_min = key.endswith('.min')
        is_range_max = key.endswith('.max')
        is_enabled_toggle = key.endswith('.enabled')

        if is_range_min or is_range_max:
            base_key = key.rsplit('.', 1)[0]
            suffix = key.rsplit('.', 1)[1] # 'min' or 'max'

            if base_key not in range_partials:
                 range_partials[base_key] = {}

            # Try converting to float or int for ranges
            try:
                val = float(value_str)
                if val.is_integer(): val = int(val)
                range_partials[base_key][suffix] = val
            except ValueError:
                 logger.warning(f"Could not parse range value for {key}: '{value_str}'")
                 continue # Skip this partial value

        elif is_enabled_toggle:
             base_key = key.rsplit('.', 1)[0]
             # Checkboxes submit 'on' when checked, or nothing when unchecked.
             # We assume if the '.enabled' key exists, it was checked.
             bool_value = (value_str.lower() == 'on')
             set_nested_value(overrides, base_key + '.enabled', bool_value) # Store as 'enabled' flag if needed by raffler
             # Also potentially set the main value based on enabled state? Depends on config structure.
             # If the config expects just the probability key, the Raffler needs to check '.enabled'.
             # For simplicity, let's assume the Raffler checks for `params.get('enabled', True)`


        else: # Handle regular keys
            value = None
            # Try conversions in order: int -> float -> bool -> string
            try:
                 value = int(value_str)
            except ValueError:
                 try:
                      value = float(value_str)
                 except ValueError:
                      # Check for boolean strings explicitly
                      if value_str.lower() in ['true', 'on', 'yes']:
                           value = True
                      elif value_str.lower() in ['false', 'off', 'no']:
                           value = False
                      else:
                           value = value_str # Keep as string if all else fails

            # Set the value in the nested overrides dictionary
            set_nested_value(overrides, key, value)

    # Process collected range partials
    for base_key, parts in range_partials.items():
        if 'min' in parts and 'max' in parts:
             # Ensure min <= max if needed? Or let config validation handle it.
             range_list = [parts['min'], parts['max']]
             set_nested_value(overrides, base_key, range_list) # Store range as a list
        else:
             logger.warning(f"Incomplete range found for {base_key}: {parts}")

    logger.debug(f"Parsed form overrides: {json.dumps(overrides, indent=2)}")
    return overrides


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

# --- Run ---
def run_webui():
    ensure_dir(app.config['OUTPUT_DIR'])
    logger.info(f"Starting Web UI. Output will be in {app.config['OUTPUT_DIR']}")
    # Use waitress or gunicorn for production instead of app.run()
    # from waitress import serve
    # serve(app, host='0.0.0.0', port=5000)
    app.run(debug=False, host='0.0.0.0', port=5000) # Turn debug off for stability with threading


# Example of running webui directly (can be called from main.py)
# if __name__ == "__main__":
#     run_webui()
