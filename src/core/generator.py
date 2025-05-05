import numpy as np
import cv2
import os
import time
import copy
import logging
import imageio
from pathlib import Path

from .utils import get_rng, ensure_dir, normalize_image, image_to_bit_depth, get_distinct_colors, parse_value, create_color_visualization
from .configuration import randomize_config_for_sample
from ..components.raffler import Raffler
from ..components.background import generate_background
from ..components.patterns import get_pattern_positions
from ..components.shapes import create_shape_mask, render_shape, draw_shape
from ..components.noise import apply_noise
from ..outputs.writers import (save_numpy, save_image_data, save_json_data,
                             save_gif_data, save_text_file, calculate_hashes)
from ..outputs.ground_truth import (generate_instance_data, generate_combined_mask,
                                  generate_defect_mask, create_overlays,
                                  add_metadata_overlay)


logger = logging.getLogger(__name__)

import numpy as np
import cv2
import os
import time
import logging
from pathlib import Path

from .utils import get_rng, ensure_dir, normalize_image, image_to_bit_depth, get_distinct_colors
from .configuration import randomize_config_for_sample
from ..components.raffler import Raffler
from ..components.background import generate_background
from ..components.patterns import get_pattern_positions
from ..components.shapes import create_shape_mask, render_shape
from ..components.artifacts.shape_level import (apply_edge_ripple, apply_breaks_holes,
                                                apply_local_elastic, apply_contour_smoothing,
                                                apply_local_brightness, apply_etch_bias,
                                                apply_local_affine, apply_shape_border)
from ..components.artifacts.geometric import apply_affine, apply_elastic
from ..components.artifacts.instrument_optical import (apply_psf_blur, apply_defocus_blur,
                                                     apply_charging, apply_topographic_shading,
                                                     apply_gradient_illumination, apply_striping_smearing,
                                                     apply_fixed_pattern_noise, apply_edge_brightness)
from ..components.noise import apply_noise
from ..outputs.writers import (save_numpy, save_image_data, save_json_data,
                             save_gif_data, save_text_file, calculate_hashes)
from ..outputs.ground_truth import (generate_instance_data, generate_combined_mask,
                                  generate_defect_mask, create_overlays,
                                  add_metadata_overlay)

# Need PerlinNoise here if generating map once per layer
try: from perlin_noise import PerlinNoise; HAS_PERLIN = True
except ImportError: HAS_PERLIN = False

logger = logging.getLogger(__name__)

# Artifacts that modify the MASK
SHAPE_MASK_ARTIFACT_FUNCS = {
    'edge_ripple': apply_edge_ripple,
    'breaks_holes': apply_breaks_holes,
    'etch_bias': apply_etch_bias, # Applied ONCE after loop now
    'local_elastic': apply_local_elastic, # If mode='local_elastic'
    'local_affine': apply_local_affine,   # If mode='local_affine'
    'contour_smoothing': apply_contour_smoothing,
}
# Artifacts that modify the RENDERED instance (float array)
SHAPE_RENDER_ARTIFACT_FUNCS = {
    'shape_border': apply_shape_border,
    'local_brightness': apply_local_brightness,
}
GEOMETRIC_ARTIFACT_FUNCS = {
    'affine': apply_affine,
    'elastic': apply_elastic,
}
INSTRUMENT_ARTIFACT_FUNCS = {
    'psf_blur': apply_psf_blur,
    'defocus_blur': apply_defocus_blur,
    'charging': apply_charging,
    'topographic_shading': apply_topographic_shading,
    'edge_brightness': apply_edge_brightness, # <-- Add new func
    'gradient_illumination': apply_gradient_illumination,
    'striping_smearing': apply_striping_smearing,
    'fixed_pattern_noise': apply_fixed_pattern_noise,
}

def generate_sample(sample_idx, sample_seed, base_config, output_parent_dir):
    """Generates a single synthetic SEM sample with all outputs."""
    
    # --- Setup unique logger for this sample ---
    sample_logger = logging.getLogger(f"semgen_sample_{sample_idx:05d}")
    sample_logger.propagate = False # Avoid duplicate logging if root is configured
    log_dir = Path(output_parent_dir) / f"sem_{sample_idx:05d}" / "logs" # Correct log dir path
    ensure_dir(log_dir)
    log_file_path = log_dir / "generation.log"
    # Add handler only if one for this file doesn't exist on this logger
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file_path) for h in sample_logger.handlers):
        log_file_handler = logging.FileHandler(log_file_path, mode='w')
        # Use a more detailed formatter for sample logs
        log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s'))
        sample_logger.addHandler(log_file_handler)
        # Inherit level from root or set explicitly
        sample_logger.setLevel(logging.INFO) # Use level set by main process/CLI
    # --- End logger setup ---

    start_time = time.time()
    sample_rng = get_rng(sample_seed)
    try:
        config = randomize_config_for_sample(base_config, sample_seed)
        artifact_raffle_settings = config.get('artifact_raffle', {})
        geom_mode = artifact_raffle_settings.get('per_instance_geometric_mode', 'none') # Default to none
        out_opts = config.get('output_options', {})
        logger.info(f"--- Generating Sample {sample_idx:05d} (Seed: {sample_seed}) (geometric mode: {geom_mode}) ---")

        # --- Safely get image settings ---
        image_settings = config.get('image_settings', {}) # Get the sub-dict safely
        h, w = image_settings.get('resolution', [256, 256]) # Default if missing
        bit_depth = image_settings.get('bit_depth', 16) # Default if missing

        # Get magnification (already randomized if range was present)
        magnification = image_settings.get('magnification', 1.0) # Default if missing

        # Get pixel size, providing a default value if missing
        pixel_size_nm_at_1x = image_settings.get('pixel_size_nm_at_1x', None) # Get safely
        pixel_size_nm = None
        if pixel_size_nm_at_1x is not None and magnification != 0:
            pixel_size_nm = pixel_size_nm_at_1x / magnification
        elif pixel_size_nm_at_1x is None:
            logger.warning(f"[Sample {sample_idx:05d}] 'pixel_size_nm_at_1x' not found.")
        elif magnification == 0: # Avoid division by zero
            logger.warning(f"[Sample {sample_idx:05d}] Magnification is zero, cannot calculate pixel size. Scale bar may be incorrect.")

        sample_name = f"sem_{sample_idx:05d}"
        sample_output_dir = Path(output_parent_dir) / sample_name
        ensure_dir(sample_output_dir)
        ensure_dir(sample_output_dir / "layers")
        ensure_dir(sample_output_dir / "layers_combined")

        # Use sample_logger for logging within this function from now on
        sample_logger.info(f"--- Generating Sample {sample_idx:05d} (Seed: {sample_seed}) ---")

        is_negative_control = sample_rng.random() < config.get('run_settings',{}).get('negative_control_probability', 0.0)
        if is_negative_control:
           sample_logger.info("Generating as Negative Control (noise/artifacts only).")
           config['layering']['selected_layers'] = []
           save_text_file("This is a negative control sample.", sample_output_dir / "negative_control_flag.txt")

        # --- Log config section before Raffler init ---
        artifact_raffle_settings = config.get('artifact_raffle', 'MISSING')
        sample_logger.debug(f"Initializing Raffler with artifact_raffle type: {type(artifact_raffle_settings)}")
        if isinstance(artifact_raffle_settings, dict):
            shape_artifacts_for_raffler = artifact_raffle_settings.get('categories',{}).get('shape', 'MISSING_SHAPE_CATEGORY')
            sample_logger.debug(f"Shape artifacts passed to Raffler type: {type(shape_artifacts_for_raffler)}")
            if isinstance(shape_artifacts_for_raffler, list):
                 names_for_raffler = [item.get('name', 'NO_NAME') if isinstance(item, dict) else 'INVALID_ITEM_TYPE' for item in shape_artifacts_for_raffler]
                 sample_logger.debug(f"Shape artifact names for Raffler: {names_for_raffler}")
            else:
                sample_logger.error(f"Shape artifacts section for Raffler is not a list: {shape_artifacts_for_raffler}")
        else:
            sample_logger.error(f"Artifact raffle section for Raffler is not dict or missing: {artifact_raffle_settings}")
        # ---

        raff = Raffler(config.get('artifact_raffle', {}), sample_rng)
        bg_conf = config.get('background', {})
        background_clean = generate_background(bg_conf, (h, w), magnification, sample_rng) # Pass RNG
        initial_background = background_clean.copy()

        image_clean = background_clean.copy()
        sample_logger.debug("Background generated.")
        
        all_layers_data = {}
        layer_instance_counts = {}
        layer_masks_original = {}
        layer_masks_actual = {}
        layer_defect_masks = {} # Store defect masks per layer
        layer_renders_actual = [] # List of actual rendered layer buffers (float32)

        instance_id_counter = 1
        layering_conf = config.get('layering', {})
        selected_layers = layering_conf.get('selected_layers', [])
        composition_mode = layering_conf.get('composition_mode', 'additive')
        randomize_layer_order = layering_conf.get('randomize_order', False) # Use randomized value

        layer_indices = list(range(len(selected_layers)))
        if randomize_layer_order: sample_rng.shuffle(layer_indices)

        layer_colors = get_distinct_colors(len(selected_layers))
        layer_id_to_color = {i: layer_colors[i] for i in range(len(selected_layers))}

        for layer_render_idx, layer_config_idx in enumerate(layer_indices):
            layer_conf = selected_layers[layer_config_idx]
            layer_id_str = f"layer_{layer_config_idx:02d}" # Use index for path
            layer_id_name = layer_conf.get('layer_id', layer_id_str) # Use name from config if present
            layer_output_dir = sample_output_dir / "layers" / layer_id_str
            ensure_dir(layer_output_dir)
            sample_logger.info(f"Processing Layer {layer_config_idx}: '{layer_id_name}'")
            intensity = layer_conf.get('intensity', 0.5) # Default to 0.5 if missing
            # --- Handle shape_choices vs shape ---
            if 'shape' in layer_conf:
                 shape_type = layer_conf.get('shape', 'circle')
            elif 'shape_choices' in layer_conf: # Check if selection was made by randomization
                 # The key should be 'shape' after randomization if _choices was used
                 shape_type = layer_conf.get('shape', 'circle') # Get randomized value
                 if isinstance(shape_type, list): # Safety check if randomization failed
                     sample_logger.warning(f"Shape type for layer {layer_id_name} is still list, choosing randomly: {shape_type}")
                     shape_type = sample_rng.choice(shape_type) if shape_type else 'circle'
            else:
                 sample_logger.warning(f"Layer {layer_id_name} missing 'shape' or 'shape_choices'. Defaulting to circle.")
                 shape_type = 'circle'
            # ---

            alpha = layer_conf.get('alpha', 1.0)
            shape_params_base = layer_conf.get('shape_params', {})
            pattern_params = layer_conf.get('pattern_params', {})
            pattern_type = layer_conf.get('pattern', 'grid')
            if 'intensity' not in layer_conf:
                sample_logger.warning(f"Layer config {layer_id_name} missing 'intensity'. Using default {intensity}. Config: {layer_conf}")
            if 'pattern' not in layer_conf:
                sample_logger.warning(f"Layer config {layer_id_name} missing 'pattern'. Using default {pattern_type}. Config: {layer_conf}")
            # ---

            target_intensity = layer_conf.get('intensity', 0.5) # Store original target intensity
            # This determines WHICH artifacts *might* be applied to instances in this layer
            layer_shape_artifacts_defs = raff.raffle_effects('shape') # Get definitions {name: ..., params: {..._range: [...]}}
            # --- Pre-generate Layer-wide Noise Map for Local Brightness (if needed) ---
            brightness_artifact_def = next((a for a in layer_shape_artifacts_defs if a['name'] == 'local_brightness'), None)
            layer_brightness_noise_map = None
            if brightness_artifact_def and HAS_PERLIN:
                try:
                    # Use average params for the layer noise map? Or the first instance's? Average is better.
                    # This assumes params has scale/contrast keys directly after randomization
                    lb_params = brightness_artifact_def.get('params', {})
                    lb_scale = lb_params.get('scale', 20.0)
                    lb_octaves = lb_params.get('octaves', 4) # Add octaves to config if needed
                    lb_contrast = lb_params.get('contrast', 0.1)

                    sample_logger.debug(f"Generating layer-wide brightness noise map (scale={lb_scale})...")
                    noise_gen = PerlinNoise(octaves=lb_octaves, seed=sample_rng.randint(0, 2**32 - 1))
                    layer_brightness_noise_map = np.zeros((h,w), dtype=np.float32)
                    for r in range(h):
                        for c in range(w):
                            layer_brightness_noise_map[r,c] = noise_gen([r/lb_scale, c/lb_scale])

                    # Normalize -1 to 1, scale by contrast (will be applied per instance)
                    mean_noise = np.mean(layer_brightness_noise_map)
                    std_noise = np.std(layer_brightness_noise_map)
                    if std_noise > 1e-6:
                       layer_brightness_noise_map = (layer_brightness_noise_map - mean_noise) / std_noise
                    else: # Flat noise map
                        layer_brightness_noise_map.fill(0.0)
                    # Don't apply contrast here, apply variation per instance later using this map
                    sample_logger.debug("Layer-wide brightness noise map generated.")

                except Exception as noise_err:
                    sample_logger.error(f"Failed to generate layer brightness noise map: {noise_err}", exc_info=True)
                    layer_brightness_noise_map = None # Ensure it's None on error
            # --- End Pre-generation ---

            # Get positions using combined params
            positions_or_paths = get_pattern_positions(pattern_type, (h, w), shape_params_base, pattern_params, sample_rng)
            is_path_based = pattern_type in ['sine_wave_horizontal'] # Add future path patterns here

            # --- Initialize Layer Accumulators ---
            layer_combined_mask_original = np.zeros((h, w), dtype=np.uint8)
            layer_combined_mask_actual = np.zeros((h, w), dtype=np.uint8)
            layer_render_buffer = np.zeros((h, w), dtype=np.float32) # Float for rendering intensity
            num_instances_in_layer = 0
            applied_shape_artifacts_list = [] # Track artifacts applied in this layer


            # Note: raff.raffle_effects already randomized the ranges into single values for the layer
            # We will use these layer-level randomized values as the *center* for per-instance variation
            sample_logger.debug(f"Layer {layer_config_idx}: Raffled shape artifacts to potentially apply: {[a['name'] for a in layer_shape_artifacts_defs]}")
            for idx, item in enumerate(positions_or_paths):
                shape_params = copy.deepcopy(shape_params_base) # Use deep copy for safety
                
                # --- Declare variables used in both branches ---
                original_mask_instance = None
                actual_mask_instance = None
                instance_render = None
                applied_mask_artifacts_instance = []
                applied_render_artifacts_instance = []

                # Branch A: Path-Based (e.g., Sine Wave)
                if is_path_based:
                    path_points = item
                    if not path_points or len(path_points) < 2: continue

                    thickness = shape_params.get('thickness', 2) # Get thickness from shape params
                    thickness = max(1, int(round(thickness)))

                    # 1. Create initial mask by drawing path
                    temp_mask = np.zeros((h, w), dtype=np.uint8)
                    pts_np = np.array([path_points], dtype=np.int32)
                    cv2.polylines(temp_mask, pts_np, isClosed=False, color=1, thickness=thickness)
                    original_mask_instance = temp_mask
                    if np.sum(original_mask_instance) == 0: continue # Skip if path outside bounds

                    # 2. Apply Mask Artifacts
                    actual_mask_instance = original_mask_instance.copy()
                    for layer_artifact_def in layer_shape_artifacts_defs:
                        artifact_name = layer_artifact_def.get('name')
                        if not artifact_name or artifact_name in SHAPE_RENDER_ARTIFACT_FUNCS or artifact_name == 'etch_bias': continue # Skip render/layer artifacts

                        # --- *** START INSERTED CODE *** ---
                        # Create Per-Instance Parameters with Variation
                        layer_params = layer_artifact_def.get('params', {})
                        instance_params = {}
                        variation_factor = 0.2 # Base variation
                        affine_variation_factor = 0.1 # Smaller variation for affine
                        current_variation = variation_factor if artifact_name != 'local_affine' else affine_variation_factor
                        for p_name, p_val in layer_params.items():
                            if isinstance(p_val, (int, float)):
                                offset = p_val * current_variation * sample_rng.uniform(-1.0, 1.0)
                                if isinstance(p_val, int): instance_params[p_name] = max(0, int(round(p_val + offset)))
                                else: instance_params[p_name] = p_val + offset
                            else: instance_params[p_name] = p_val # Keep non-numeric

                        # Check Mode & Get Function for Geometric Artifacts
                        apply_this_artifact = True
                        target_mask_artifact_func = None
                        if artifact_name == 'local_elastic':
                            if geom_mode != 'local_elastic': apply_this_artifact = False
                            else: target_mask_artifact_func = SHAPE_MASK_ARTIFACT_FUNCS.get(artifact_name)
                        elif artifact_name == 'local_affine':
                            if geom_mode != 'local_affine': apply_this_artifact = False
                            else: target_mask_artifact_func = SHAPE_MASK_ARTIFACT_FUNCS.get(artifact_name)
                        elif artifact_name in SHAPE_MASK_ARTIFACT_FUNCS: # Other mask artifacts
                              target_mask_artifact_func = SHAPE_MASK_ARTIFACT_FUNCS.get(artifact_name)
                        else: apply_this_artifact = False # Not a known mask artifact
                        # --- *** END INSERTED CODE *** ---

                        if apply_this_artifact and target_mask_artifact_func:
                            try:
                                actual_mask_instance = target_mask_artifact_func(actual_mask_instance, instance_params, sample_rng)
                                applied_mask_artifacts_instance.append(artifact_name)
                            except Exception as e: sample_logger.error(f"Error applying mask artifact {artifact_name} to path instance {idx}: {e}", exc_info=True)

                    # 3. Initial Render (using final mask)
                    instance_render = np.zeros_like(layer_render_buffer)
                    instance_render[actual_mask_instance > 0] = intensity * alpha # Apply intensity

                    # 4. Apply Render Artifacts
                    render_artifact_order = ['shape_border', 'local_brightness']
                    for artifact_name in render_artifact_order:
                          render_artifact_def = next((a for a in layer_shape_artifacts_defs if a.get('name') == artifact_name), None)
                          if render_artifact_def:
                               target_render_artifact_func = SHAPE_RENDER_ARTIFACT_FUNCS.get(artifact_name)
                               if target_render_artifact_func:
                                    try:
                                        # *** Re-vary params specific to this render artifact ***
                                        layer_params_render = render_artifact_def.get('params', {})
                                        instance_params_render = {}
                                        variation_factor = 0.2 # Use appropriate variation
                                        for p_name, p_val in layer_params_render.items():
                                            if isinstance(p_val, (int, float)):
                                                offset = p_val * variation_factor * sample_rng.uniform(-1.0, 1.0)
                                                if isinstance(p_val, int): instance_params_render[p_name] = max(1, int(round(p_val + offset))) # Ensure thickness > 0 for border
                                                else: instance_params_render[p_name] = p_val + offset
                                            else: instance_params_render[p_name] = p_val
                                        # ***

                                        if artifact_name == 'shape_border':
                                            instance_render = target_render_artifact_func(instance_render, actual_mask_instance, instance_params_render, intensity, sample_rng)
                                        elif artifact_name == 'local_brightness':
                                            if layer_brightness_noise_map is not None:
                                                instance_contrast = instance_params_render.get('contrast', 0.1)
                                                noise_slice = layer_brightness_noise_map[actual_mask_instance > 0]
                                                brightness_variation = noise_slice * instance_contrast
                                                instance_render[actual_mask_instance > 0] *= (1.0 + brightness_variation)
                                                instance_render = np.clip(instance_render, 0.0, 1.0)
                                            else: sample_logger.warning("Skipping local_brightness (no noise map).")
                                        applied_render_artifacts_instance.append(artifact_name)
                                    except Exception as e: sample_logger.error(f"Error applying render artifact {artifact_name} to path instance {idx}: {e}", exc_info=True)

                # Branch B: Position-Based (Existing Shapes)
                else:
                    pos = item
                    # --- Set shape params based on position type ---
                    if shape_type.endswith('line') and isinstance(pos, tuple) and len(pos) == 2 and isinstance(pos[0], tuple):
                         shape_params['x1'], shape_params['y1'] = pos[0]
                         shape_params['x2'], shape_params['y2'] = pos[1]
                    elif isinstance(pos, tuple) and len(pos) == 2 and isinstance(pos[0], (int, float)):
                         shape_params['center_x'], shape_params['center_y'] = pos
                    else:
                         sample_logger.warning(f"Unsupported position format for shape {shape_type}: {pos}. Skipping instance.")
                         continue

                    # 1. Generate Original Mask
                    original_mask_instance = create_shape_mask(shape_type, shape_params, (h, w), rng=sample_rng)
                    if np.sum(original_mask_instance) == 0: continue

                    # 2. Apply MASK-MODIFYING Artifacts
                    actual_mask_instance = original_mask_instance.copy()
                    for layer_artifact_def in layer_shape_artifacts_defs:
                        artifact_name = layer_artifact_def.get('name')
                        if not artifact_name or artifact_name in SHAPE_RENDER_ARTIFACT_FUNCS or artifact_name == 'etch_bias': continue

                        # --- *** START INSERTED CODE *** ---
                        # Create Per-Instance Parameters with Variation
                        layer_params = layer_artifact_def.get('params', {})
                        instance_params = {}
                        variation_factor = 0.2; affine_variation_factor = 0.1
                        current_variation = variation_factor if artifact_name != 'local_affine' else affine_variation_factor
                        for p_name, p_val in layer_params.items():
                            if isinstance(p_val, (int, float)):
                                offset = p_val * current_variation * sample_rng.uniform(-1.0, 1.0)
                                if isinstance(p_val, int): instance_params[p_name] = max(0, int(round(p_val + offset)))
                                else: instance_params[p_name] = p_val + offset
                            else: instance_params[p_name] = p_val

                        # Check Mode & Get Function for Geometric Artifacts
                        apply_this_artifact = True
                        target_mask_artifact_func = None
                        if artifact_name == 'local_elastic':
                            if geom_mode != 'local_elastic': apply_this_artifact = False
                            else: target_mask_artifact_func = SHAPE_MASK_ARTIFACT_FUNCS.get(artifact_name)
                        elif artifact_name == 'local_affine':
                            if geom_mode != 'local_affine': apply_this_artifact = False
                            else: target_mask_artifact_func = SHAPE_MASK_ARTIFACT_FUNCS.get(artifact_name)
                        elif artifact_name in SHAPE_MASK_ARTIFACT_FUNCS:
                            target_mask_artifact_func = SHAPE_MASK_ARTIFACT_FUNCS.get(artifact_name)
                        else: apply_this_artifact = False
                        # --- *** END INSERTED CODE *** ---

                        if apply_this_artifact and target_mask_artifact_func:
                            try:
                                actual_mask_instance = target_mask_artifact_func(actual_mask_instance, instance_params, sample_rng)
                                applied_mask_artifacts_instance.append(artifact_name)
                            except Exception as e: sample_logger.error(f"Error applying mask artifact {artifact_name} to instance {idx}: {e}", exc_info=True)

                    # 3. Initial Render (using shape params and final mask)
                    instance_render = np.zeros_like(layer_render_buffer)
                    # Render the shape *without* intensity first, potentially with AA
                    # Use original shape_params for rendering geometry
                    temp_render_geometry = render_shape(shape_type, shape_params, (h, w), intensity=1.0, anti_aliasing=True, rng=sample_rng)
                    # Apply intensity only where the final *actual* mask is set
                    instance_render[actual_mask_instance > 0] = temp_render_geometry[actual_mask_instance > 0] * intensity * alpha
                    instance_render = np.clip(instance_render, 0.0, 1.0)


                    # 4. Apply RENDER-MODIFYING Artifacts
                    render_artifact_order = ['shape_border', 'local_brightness']
                    for artifact_name in render_artifact_order:
                        render_artifact_def = next((a for a in layer_shape_artifacts_defs if a.get('name') == artifact_name), None)
                        if render_artifact_def:
                            target_render_artifact_func = SHAPE_RENDER_ARTIFACT_FUNCS.get(artifact_name)
                            if target_render_artifact_func:
                                try:
                                    # *** Re-vary params specific to this render artifact ***
                                    layer_params_render = render_artifact_def.get('params', {})
                                    instance_params_render = {}
                                    variation_factor = 0.2 # Use appropriate variation
                                    for p_name, p_val in layer_params_render.items():
                                        if isinstance(p_val, (int, float)):
                                            offset = p_val * variation_factor * sample_rng.uniform(-1.0, 1.0)
                                            if isinstance(p_val, int): instance_params_render[p_name] = max(1, int(round(p_val + offset))) # Ensure thickness > 0
                                            else: instance_params_render[p_name] = p_val + offset
                                        else: instance_params_render[p_name] = p_val
                                    # ***

                                    if artifact_name == 'shape_border':
                                        instance_render = target_render_artifact_func(instance_render, actual_mask_instance, instance_params, intensity, sample_rng)
                                    elif artifact_name == 'local_brightness':
                                        if layer_brightness_noise_map is not None:
                                            instance_contrast = instance_params.get('contrast', 0.1)
                                            noise_slice = layer_brightness_noise_map[actual_mask_instance > 0]
                                            brightness_variation = noise_slice * instance_contrast
                                            instance_render[actual_mask_instance > 0] *= (1.0 + brightness_variation)
                                            instance_render = np.clip(instance_render, 0.0, 1.0)
                                        else: sample_logger.warning("Skipping local_brightness (no noise map).")
                                    applied_render_artifacts_instance.append(artifact_name)
                                except Exception as e: sample_logger.error(f"Error applying render artifact {artifact_name} to instance {idx}: {e}", exc_info=True)

                # --- C. Common Instance Processing ---
                if original_mask_instance is not None and actual_mask_instance is not None and instance_render is not None:
                    layer_combined_mask_original |= original_mask_instance
                    layer_combined_mask_actual |= actual_mask_instance
                    layer_render_buffer += instance_render # Add final instance render
                    # Track applied artifacts
                    applied_shape_artifacts_list.extend(applied_mask_artifacts_instance)
                    applied_shape_artifacts_list.extend(applied_render_artifacts_instance)
                    num_instances_in_layer += 1
                # --- End Common Processing ---

            print('#######################8##########################')

            # Clip layer render buffer after all instances are added
            layer_render_buffer = np.clip(layer_render_buffer, 0.0, 1.0)
            layer_renders_actual.append(layer_render_buffer)

            # Generate layer defect mask
            defect_mask_layer = generate_defect_mask(layer_combined_mask_original, layer_combined_mask_actual)

            all_layers_data[layer_config_idx] = {
                'id': layer_id_name,
                'original_mask': layer_combined_mask_original,
                'actual_mask': layer_combined_mask_actual,
                'defect_mask': defect_mask_layer,
                'render': layer_render_buffer.copy() if out_opts.get('save_per_layer_renders') else None, # Store if saving needed
                'output_dir': layer_output_dir,
                'num_instances': num_instances_in_layer,
                'applied_shape_artifacts': list(set(applied_shape_artifacts_list)) # Unique list
            }
            layer_masks_original[layer_config_idx] = layer_combined_mask_original
            layer_masks_actual[layer_config_idx] = layer_combined_mask_actual
            if defect_mask_layer is not None:
                layer_defect_masks[layer_config_idx] = defect_mask_layer
            layer_instance_counts[layer_config_idx] = num_instances_in_layer

            # --- Save Per-Layer Outputs ---
            if out_opts['save_masks']:
                save_numpy(layer_combined_mask_original, layer_output_dir / "original_mask.npy")
                save_numpy(layer_combined_mask_actual, layer_output_dir / "actual_mask.npy")
                if out_opts['save_defect_masks'] and defect_mask_layer is not None:
                     save_numpy(defect_mask_layer, layer_output_dir / "defect_mask.npy")

            if out_opts['save_visualizations']:
                save_image_data(layer_combined_mask_original.astype(float), layer_output_dir / "original_mask_vis.png", 8)
                save_image_data(layer_combined_mask_actual.astype(float), layer_output_dir / "actual_mask_vis.png", 8)
                if out_opts['save_defect_masks'] and defect_mask_layer is not None:
                    save_image_data(defect_mask_layer.astype(float), layer_output_dir / "defect_mask_vis.png", 8)

            if out_opts.get('save_per_layer_renders') and all_layers_data[layer_config_idx]['render'] is not None:
                save_image_data(all_layers_data[layer_config_idx]['render'], layer_output_dir / "render_actual_vis.png", bit_depth)

        sample_logger.debug("Finished layer loop.")
        # 5. Compose Layers
        sample_logger.info(f"Composing {len(layer_renders_actual)} layers using mode: {composition_mode}")
        cumulative_layers_for_gif = [initial_background.copy()]
        for layer_buffer in layer_renders_actual:
            if composition_mode == 'additive':
                image_clean += layer_buffer
            elif composition_mode == 'multiplicative':
                image_clean *= (1.0 + layer_buffer * 2) # Example multiplicative blend
            elif composition_mode == 'overwrite':
                # Need the mask for the current layer buffer being applied
                # This requires matching buffer back to layer config - requires careful indexing
                # Let's find the corresponding mask using the render order index
                current_layer_config_idx = layer_indices[len(cumulative_layers_for_gif)-1] # Index of layer config for this buffer
                mask = all_layers_data[current_layer_config_idx]['actual_mask']
                image_clean[mask > 0] = layer_buffer[mask > 0]
            else:
                image_clean += layer_buffer # Default additive
            image_clean = np.clip(image_clean, 0.0, 1.0) # Clip after each composition
            cumulative_layers_for_gif.append(image_clean.copy())


        image_clean_pre_warp = image_clean.copy()
        sample_logger.debug("Layers composed.")
        # 6. Generate Combined/Instance Masks (Pre-Warp)
        combined_mask_original_pre_warp = generate_combined_mask(layer_masks_original)
        combined_mask_actual_pre_warp = generate_combined_mask(layer_masks_actual)
        instance_mask_pre_warp, instance_meta = generate_instance_data(layer_masks_actual, layer_instance_counts)
        sample_logger.debug("Global geometric artifacts applied.")


        # 7. Apply Global Artifacts (Geometric, Instrument) - Use Oversized Canvas
        sample_logger.info("Applying global artifacts...")
        margin_factor = config['image_settings']['oversize_margin_factor']
        margin_h, margin_w = int(h * margin_factor), int(w * margin_factor)
        oversized_h, oversized_w = h + 2 * margin_h, w + 2 * margin_w
        oversized_shape = (oversized_h, oversized_w)

        def embed_in_oversized(img, target_shape, margin_h, margin_w, border_mode=cv2.BORDER_REFLECT_101, border_value=0):
            if img is None: return None
            # Use copyMakeBorder for cleaner padding
            padded = cv2.copyMakeBorder(img, margin_h, margin_h, margin_w, margin_w, border_mode, value=border_value)
            # Ensure exact shape due to potential rounding issues
            return padded[:target_shape[0], :target_shape[1]]

        image_clean_oversized = embed_in_oversized(image_clean, oversized_shape, margin_h, margin_w)
        # --- Semantic Map Setup ---
        semantic_map_pre_warp = None
        if out_opts.get('save_semantic_map', False):
             semantic_map_pre_warp = np.zeros((h, w), dtype=np.uint8) # Use uint8 if < 255 layers
             # Re-compose layers onto semantic map respecting order/mode (simplification: overwrite)
             for layer_render_idx_sem, layer_config_idx_sem in enumerate(layer_indices):
                  mask_sem = layer_masks_actual.get(layer_config_idx_sem)
                  if mask_sem is not None:
                       semantic_map_pre_warp[mask_sem > 0] = layer_config_idx_sem + 1 # Layer index + 1
             semantic_map_oversized = embed_in_oversized(semantic_map_pre_warp, oversized_shape, margin_h, margin_w, cv2.BORDER_CONSTANT, 0)
             logger.debug("Prepared semantic map for warping.")
        else:
             semantic_map_oversized = None
        # --- End Semantic Map Setup ---

        
        combined_actual_mask_oversized = embed_in_oversized(combined_mask_actual_pre_warp, oversized_shape, margin_h, margin_w, cv2.BORDER_CONSTANT, 0)
        instance_mask_oversized = embed_in_oversized(instance_mask_pre_warp, oversized_shape, margin_h, margin_w, cv2.BORDER_CONSTANT, 0)

        # Apply Geometric Warps
        geometric_artifacts = raff.raffle_effects('geometric')
        warp_field_combined = None
        masks_to_warp = [m for m in [semantic_map_oversized] if m is not None] # Only warp semantic map now
        applied_geometric_artifacts = [] # Track applied artifacts

        for artifact in geometric_artifacts:
             func = GEOMETRIC_ARTIFACT_FUNCS.get(artifact['name'])
             if func:
                  try:
                       sample_logger.info(f"Applying geometric artifact: {artifact['name']}")
                       # Pass image and semantic map (if present)
                       image_clean_oversized, warped_masks_out, warp_field = \
                           func(image_clean_oversized, masks_to_warp, artifact['params'], sample_rng)
                       if warped_masks_out: masks_to_warp = warped_masks_out # Update for next warp step
                       if warp_field is not None: warp_field_combined = warp_field
                       applied_geometric_artifacts.append(artifact['name'])
                  except Exception as e:
                      sample_logger.error(f"Error applying geometric artifact {artifact['name']}: {e}", exc_info=True)
             else:
                 sample_logger.warning(f"Geometric artifact function '{artifact['name']}' not found.")

        # Update main mask variables after warping
        if masks_to_warp:
             semantic_map_oversized = masks_to_warp[0]

        # Center Crop Back
        image_clean_warped = image_clean_oversized[margin_h:margin_h+h, margin_w:margin_w+w]
        combined_actual_mask = combined_actual_mask_oversized[margin_h:margin_h+h, margin_w:margin_w+w] if combined_actual_mask_oversized is not None else None
        semantic_map = semantic_map_oversized[margin_h:margin_h+h, margin_w:margin_w+w] if semantic_map_oversized is not None else None
        warp_field_final = warp_field_combined[margin_h:margin_h+h, margin_w:margin_w+w] if warp_field_combined is not None else None
        instance_mask = instance_mask_oversized[margin_h:margin_h+h, margin_w:margin_w+w] if instance_mask_oversized is not None else None
        # --- Save post-geometric warp intermediate if requested ---
        image_post_geometric_warp = image_clean_warped.copy() # Capture state here
        if out_opts.get('save_extra_intermediate', False):
             path_post_geo = sample_output_dir / "image_post_geometric_warp.png"
             save_image_data(image_post_geometric_warp, path_post_geo, 8)
             # No need to add to metadata paths automatically, handled later if needed


        # Apply Instrument Artifacts
        instrument_artifacts = raff.raffle_effects('instrument')
        image_post_instrument = image_post_geometric_warp # Start from post-warp
        applied_instrument_artifacts = []
        total_added_fpn = np.zeros_like(image_post_instrument)
        topography_height_map = None # Initialize


        for artifact in instrument_artifacts:
            func = INSTRUMENT_ARTIFACT_FUNCS.get(artifact['name'])
            if func:
                try:
                    sample_logger.info(f"Applying instrument artifact: {artifact['name']}")
                    if artifact['name'] == 'topographic_shading':
                         # Get height map back
                         image_post_instrument, height_map_generated = func(
                             image_post_instrument, all_layers_data, artifact['params'], sample_rng
                         )
                         # Store height map if saving is enabled
                         if out_opts.get('save_topography_height_map', False):
                              topography_height_map = height_map_generated
                    elif artifact['name'] == 'fixed_pattern_noise':
                         image_post_instrument, fpn_map = func(image_post_instrument, artifact['params'], sample_rng)
                         total_added_fpn += fpn_map
                    elif artifact['name'] == 'edge_brightness':
                        # Needs the image *after* topography/blur ideally, apply near end?
                        # Or apply here? Let's apply here for now.
                         image_post_instrument = func(image_post_instrument, artifact['params'], sample_rng)
                    else: # Other instrument effects
                         image_post_instrument = func(image_post_instrument, artifact['params'], sample_rng)

                    applied_instrument_artifacts.append(artifact['name'])
                except Exception as e:
                    sample_logger.error(f"Error applying instrument artifact {artifact['name']}: {e}", exc_info=True)
            else: sample_logger.warning(f"Instrument artifact function '{artifact['name']}' not found.")

        image_post_instrument = np.clip(image_post_instrument, 0.0, 1.0)

        sample_logger.debug("Instrument artifacts applied.")
        # --- Save post-instrument intermediate if requested ---
        if out_opts.get('save_extra_intermediate', False):
             path_post_inst = sample_output_dir / "image_post_instrument.png"
             save_image_data(image_post_instrument, path_post_inst, 8)


        # 8. Apply Detector Noise
        sample_logger.info("Applying detector noise...")
        image_final_noisy = image_post_instrument.copy()
        # Start with FPN map if generated, otherwise zeros
        total_added_noise = total_added_fpn.copy()
        image_final_noisy = np.clip(image_final_noisy, 0.0, 1.0)

        # --- Generate FINAL Instance/Combined Masks AFTER warp ---
        # Re-generate actual masks for layers AFTER geometric warp if needed for precise final GT
        # This is complex. Simpler: Warp the pre-warp combined/instance masks if available?
        # Let's generate final instance data based on the *warped* semantic map (or other warped masks)
        # For simplicity, let's skip precise final instance generation for now and focus on saving what we have.
        # We will save the *warped* semantic map. BBoxes/centroids will be relative to the *warped* image.

        # --- Generate Final Instance Data (using placeholder logic for now) ---
        # Ideally, we'd warp layer_masks_actual and run generate_instance_data on those.
        # Placeholder: Generate from warped semantic map? Might merge instances incorrectly.
        # Let's skip final GT generation for now to avoid complexity.
        # We will save the warped semantic map and add bbox placeholder to metadata.
        final_instance_mask = None # Placeholder
        final_instance_metadata = {} # Placeholder
        if out_opts.get('save_bounding_boxes', False):
             final_instance_metadata['placeholder'] = "Instance bbox/centroid generation after warp not fully implemented yet."


        noise_artifacts = raff.raffle_effects('noise')
        applied_noise_artifacts = []

        # Apply quantization first if present
        quant_artifact = next((a for a in noise_artifacts if a['name'] == 'quantization'), None)
        if quant_artifact:
             try:
                  sample_logger.info(f"Applying noise: {quant_artifact['name']}")
                  image_final_noisy, q_noise = apply_noise(image_final_noisy, quant_artifact['name'], quant_artifact['params'], sample_rng)
                  total_added_noise += q_noise
                  applied_noise_artifacts.append(quant_artifact['name'])
             except Exception as e:
                  sample_logger.error(f"Error applying quantization: {e}", exc_info=True)
    
        # Apply other noise types
        for artifact in noise_artifacts:
            if artifact['name'] == 'quantization': continue # Already applied
            noise_type = artifact['name']
            try:
                 sample_logger.info(f"Applying noise: {noise_type}")
                 image_final_noisy, added_noise = apply_noise(image_final_noisy, noise_type, artifact['params'], sample_rng)
                 if added_noise is not None: total_added_noise += added_noise
                 applied_noise_artifacts.append(noise_type)
            except Exception as e:
                 sample_logger.error(f"Error applying noise {noise_type}: {e}", exc_info=True)
    
        image_final_noisy = np.clip(image_final_noisy, 0.0, 1.0)
        sample_logger.debug("Noise applied.")
    
        # 9. Generate Overlays and Final Visualizations
        sample_logger.info("Generating overlays and visualizations...")
        final_image_vis_8bit = image_to_bit_depth(image_final_noisy, 8) # Use 8-bit for overlays
    
        overlay_contour_vis, instance_mask_vis, warp_field_vis = create_overlays(
            final_image_vis_8bit,
            None, # Pass None for combined mask if not generated post-warp
            final_instance_mask, # Pass final instance mask (currently None)
            warp_field_final,
            layer_id_to_color
        )
        metadata_text = f"Sample: {sample_idx:05d}\nSeed: {sample_seed}\nMag: {magnification:.2f}x"
        overlay_metadata_vis = add_metadata_overlay(final_image_vis_8bit, text=metadata_text, pixel_size_nm=pixel_size_nm)
    
        actual_layer_gif_frames = []
        if out_opts['save_gifs'] and cumulative_layers_for_gif:
            for frame in cumulative_layers_for_gif:
                 actual_layer_gif_frames.append(image_to_bit_depth(frame, 8))
    
    
        # 10. Prepare Metadata
        metadata = {
            "sample_index": sample_idx, "sample_name": sample_name, "seed": sample_seed,
            "resolution": config['image_settings']['resolution'],
            "magnification": magnification,
            "pixel_size_nm_at_1x": pixel_size_nm_at_1x,
            "pixel_size_nm_calculated": pixel_size_nm,
            "background_type": config.get('background',{}).get('selected_type'),
            "layers": [{'config_idx': idx, 'layer_id': data['id']} for idx, data in all_layers_data.items()],
            "num_layers": len(selected_layers), "composition_mode": composition_mode,
            "applied_shape_artifacts": list(set(a for data in all_layers_data.values() for a in data['applied_shape_artifacts'])),
            "applied_geometric_artifacts": applied_geometric_artifacts, # Use tracked list
            "applied_instrument_artifacts": applied_instrument_artifacts,
            "applied_noise_artifacts": applied_noise_artifacts,
            "instance_annotations": final_instance_metadata if out_opts.get('save_bounding_boxes', False) else None,
            "instance_info": instance_meta,
            "generation_time_sec": round(time.time() - start_time, 2),
            "output_paths": {}
        }
    
        # Remove instance_annotations key if None
        if metadata["instance_annotations"] is None:
            del metadata["instance_annotations"]

        # 11. Save Outputs
        sample_logger.info("Saving outputs...")
        output_paths = {}
        save_paths_list = [] # Keep track of files for hashing
    
        def add_path(key, path_obj):
             relative_path = str(path_obj.relative_to(output_parent_dir))
             output_paths[key] = relative_path
             save_paths_list.append(path_obj)
    
        # --- Main Image ---
    
        final_image_format = out_opts.get('output_formats', {}).get('final_image', 'tif')
        final_img_path = Path(output_parent_dir) / f"{sample_name}.{final_image_format}"
        save_image_data(image_final_noisy, final_img_path, bit_depth)
        add_path('final_image', final_img_path)
    
        # --- Intermediate Images ---
        if out_opts['save_intermediate']:
            path = sample_output_dir / "image_clean_pre_warp.png"
            save_image_data(image_clean_pre_warp, path, 8) # Save as 8-bit PNG vis
            output_paths['image_clean_pre_warp'] = str(path.relative_to(output_parent_dir))
    
            path = sample_output_dir / "image_clean_warped.png"
            save_image_data(image_clean_warped, path, 8)
            output_paths['image_clean_warped'] = str(path.relative_to(output_parent_dir))
    
            path = sample_output_dir / "image_post_instrument.png"
            save_image_data(image_post_instrument, path, 8)
            output_paths['image_post_instrument'] = str(path.relative_to(output_parent_dir))
    
            path = sample_output_dir / "image_final_noisy_vis.png" # Save 8-bit vis too
            save_image_data(image_final_noisy, path, 8)
            output_paths['image_final_noisy_vis'] = str(path.relative_to(output_parent_dir))
    
            # Save initial background
            path = sample_output_dir / "layers_combined" / "background.npy"
            save_numpy(initial_background, path)
            output_paths['background_npy'] = str(path.relative_to(output_parent_dir))
            path = sample_output_dir / "layers_combined" / "background_vis.png"
            save_image_data(initial_background, path, 8)
            output_paths['background_vis'] = str(path.relative_to(output_parent_dir))
    
    
        # --- Masks ---
        if out_opts['save_masks']:
            # Combined Original (Pre-Warp)
            if combined_mask_original_pre_warp is not None:
                path = sample_output_dir / "combined_original_mask.npy"
                save_numpy(combined_mask_original_pre_warp, path)
                output_paths['combined_original_mask_npy'] = str(path.relative_to(output_parent_dir))
                if out_opts['save_visualizations']:
                     path_vis = sample_output_dir / "combined_original_mask_vis.png"
                     save_image_data(combined_mask_original_pre_warp.astype(float), path_vis, 8)
                     output_paths['combined_original_mask_vis'] = str(path_vis.relative_to(output_parent_dir))
    
            # Combined Actual (Post-Warp)
            if combined_actual_mask is not None:
                path = sample_output_dir / "combined_actual_mask.npy"
                save_numpy(combined_actual_mask, path)
                output_paths['combined_actual_mask_npy'] = str(path.relative_to(output_parent_dir))
                if out_opts['save_visualizations']:
                     path_vis = sample_output_dir / "combined_actual_mask_vis.png"
                     save_image_data(combined_actual_mask.astype(float), path_vis, 8)
                     output_paths['combined_actual_mask_vis'] = str(path_vis.relative_to(output_parent_dir))
    
            # Instance Mask (Post-Warp)
            if instance_mask is not None and out_opts['save_masks']: # Check save_masks flag
                inst_mask_format = out_opts.get('output_formats', {}).get('instance_mask', 'tif').lower()
                path = sample_output_dir / f"instance_mask.{inst_mask_format}"
                saved_successfully = False
                try:
                    if inst_mask_format == 'tif' or inst_mask_format == 'tiff':
                        # No normalization needed for instance masks
                        save_image_data(instance_mask, path, bit_depth=32 if instance_mask.dtype==np.uint32 else 16, format_hint='TIFF')
                        saved_successfully = path.is_file() # Check if file was actually created
                    elif inst_mask_format == 'png':
                        # PNG usually needs 8 or 16 bit, might lose instance IDs if > 65535
                        if np.max(instance_mask) > 65535:
                            sample_logger.warning("Max instance ID > 65535, saving as 16-bit PNG might lose IDs. Consider TIF or NPY.")
                            img_to_save = instance_mask.astype(np.uint16)
                        else:
                            img_to_save = instance_mask.astype(np.uint16 if np.max(instance_mask)>255 else np.uint8)
                        save_image_data(img_to_save, path, bit_depth=16 if img_to_save.dtype==np.uint16 else 8, format_hint='PNG')
                        saved_successfully = path.is_file()
                    else:
                        sample_logger.warning(f"Unsupported instance mask format '{inst_mask_format}'. Saving as NPY.")
    
                except Exception as e:
                    sample_logger.error(f"Error saving instance mask as {inst_mask_format}: {e}", exc_info=True)
    
                # Fallback to NPY if specified format failed or wasn't image format
                if not saved_successfully:
                    sample_logger.warning(f"Failed to save instance mask as {inst_mask_format}. Saving as NPY.")
                    path = sample_output_dir / "instance_mask.npy"
                    save_numpy(instance_mask, path)
    
                # Add the final path (either image format or npy)
                add_path('instance_mask', path)
    
                # Save visualization (if needed)
                if out_opts['save_visualizations'] and instance_mask_vis is not None:
                    path_vis = sample_output_dir / "instance_mask_vis.png"
                    save_image_data(instance_mask_vis, path_vis, 8, format_hint='PNG') # Vis is usually uint8 BGR
                    add_path('instance_mask_vis', path_vis)
    
    
        if out_opts['save_masks'] and out_opts.get('save_defect_masks'):
             # Combined defect mask
             combined_defect = generate_defect_mask(combined_mask_original_pre_warp, combined_mask_actual_pre_warp) # Using pre-warp masks? Or post-warp? Pre-warp more about shape artifacts.
             if combined_defect is not None:
                  path = sample_output_dir / "combined_defect_mask.npy"
                  save_numpy(combined_defect, path)
                  add_path('combined_defect_mask_npy', path)
                  if out_opts['save_visualizations']:
                       path_vis = sample_output_dir / "combined_defect_mask_vis.png"
                       save_image_data(combined_defect.astype(float), path_vis, 8)
                       add_path('combined_defect_mask_vis', path_vis)
    
        # --- Artifact Maps ---
        if out_opts['save_warp_field'] and warp_field_final is not None:
            path = sample_output_dir / "warp_field.npy"
            save_numpy(warp_field_final, path)
            add_path('warp_field_npy', path)
            if out_opts['save_visualizations'] and warp_field_vis is not None:
                 path_vis = sample_output_dir / "warp_field_vis.png"
                 save_image_data(warp_field_vis, path_vis, 8) # Assumes BGR uint8
                 add_path('warp_field_vis', path_vis)
    
    
        if out_opts['save_noise_map'] and total_added_noise is not None:
             path = sample_output_dir / "noise_map_added.npy"
             save_numpy(total_added_noise, path)
             output_paths['noise_map_added_npy'] = str(path.relative_to(output_parent_dir))
             if out_opts['save_visualizations']:
                  path_vis = sample_output_dir / "noise_map_added_vis.png"
                  # Normalize noise for visualization (e.g., map range to 0-1)
                  save_image_data(normalize_image(total_added_noise), path_vis, 8)
                  output_paths['noise_map_added_vis'] = str(path_vis.relative_to(output_parent_dir))
    
    
        # --- Save Semantic Map ---
        if out_opts.get('save_semantic_map', False) and semantic_map is not None:
            path_sem = sample_output_dir / "semantic_map_layer_idx.npy"
            save_numpy(semantic_map.astype(np.uint8), path_sem) # Save as uint8 npy
            add_path('semantic_map_npy', path_sem)
            if out_opts['save_visualizations']:
                # Create color visualization based on layer index
                max_layer_idx = np.max(semantic_map)
                sem_colormap = {i: layer_id_to_color.get(i-1, (128,128,128)) for i in range(1, max_layer_idx + 1)} # Map index+1 to color
                sem_vis = create_color_visualization(semantic_map, sem_colormap)
                path_sem_vis = sample_output_dir / "semantic_map_layer_idx_vis.png"
                save_image_data(sem_vis, path_sem_vis, 8, format_hint='PNG')
                add_path('semantic_map_vis', path_sem_vis)

        # --- Save Height Map ---
        if out_opts.get('save_topography_height_map', False) and topography_height_map is not None:
             path_hm = sample_output_dir / "topography_height_map.npy"
             save_numpy(topography_height_map, path_hm) # Save as float npy
             add_path('height_map_npy', path_hm)
             if out_opts['save_visualizations']:
                  # Normalize for visualization
                  hm_vis = normalize_image(topography_height_map)
                  path_hm_vis = sample_output_dir / "topography_height_map_vis.png"
                  save_image_data(hm_vis, path_hm_vis, 8, format_hint='PNG')
                  add_path('height_map_vis', path_hm_vis)

        # --- Overlays & GIFs ---
        if out_opts['save_overlays']:
             if overlay_contour_vis is not None:
                 path = sample_output_dir / "overlay_contour.png"
                 imageio.imwrite(path, overlay_contour_vis)
                 output_paths['overlay_contour'] = str(path.relative_to(output_parent_dir))
             if overlay_metadata_vis is not None:
                 path = sample_output_dir / "overlay_metadata.png"
                 imageio.imwrite(path, overlay_metadata_vis)
                 output_paths['overlay_metadata'] = str(path.relative_to(output_parent_dir))
    
        if out_opts['save_gifs'] and actual_layer_gif_frames:
             path = sample_output_dir / "layers_combined" / "layers_actual_masks_color.gif"
             save_gif_data(actual_layer_gif_frames, path, duration=0.5) # Adjust duration
             output_paths['layers_actual_gif'] = str(path.relative_to(output_parent_dir))
    
    
        # --- Config & Metadata ---
        if out_opts['save_config']:
             path = sample_output_dir / "configuration.json"
             save_json_data(config, path)
             add_path('configuration', path)
    
        # --- Final Metadata Update and Save ---
        metadata['output_paths'] = output_paths
        meta_path = sample_output_dir / "metadata.json"
        save_json_data(metadata, meta_path)
        # Don't add meta_path itself to save_paths_list before hashing it
    
    
        # --- Hashes ---
        if out_opts['save_hashes']:
             hashes = calculate_hashes([p for p in save_paths_list if p.is_file()])
             hashes_path = sample_output_dir / "hashes.json"
             save_json_data(hashes, hashes_path)
    
    
        # Close sample-specific log handler
        sample_logger.removeHandler(log_file_handler)
        log_file_handler.close()
    
        sample_logger.info(f"--- Finished Sample {sample_idx:05d} in {time.time() - start_time:.2f} seconds ---")
    
        # --- Cleanup Handler ---
        # Important to close handlers opened by this process
        for handler in sample_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                 handler.close()
            sample_logger.removeHandler(handler)
        # ---
    
        return True # Indicate success
    except Exception as e:
        # Log error to the *sample-specific* log file
        sample_logger.error(f"!!! CRITICAL ERROR generating sample {sample_idx:05d} !!! Type: {type(e).__name__}, Error: {e}", exc_info=True)
        # --- Close Handler on Error Too ---
        for handler in sample_logger.handlers[:]: # Iterate copy
             if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_file_path):
                  handler.close()
                  sample_logger.removeHandler(handler)
        # ---
        return False # Indicate failure

