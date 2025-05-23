import numpy as np
import cv2
import os
import time
import copy
import logging
import imageio
import json
from pathlib import Path

from .utils import (get_rng, ensure_dir, normalize_image, image_to_bit_depth,
                    get_distinct_colors, create_color_visualization, parse_value)
from .constants import SHAPE_TYPE_MAP, NUM_SHAPE_CLASSES

from .configuration import randomize_config_for_sample
from ..components.background import generate_background
from ..components.patterns import get_pattern_positions
from ..components.shapes import create_shape_mask, render_shape
from ..components.raffler import Raffler
from ..components.noise import apply_noise
from ..components.artifacts.shape_level import (apply_edge_ripple, apply_breaks_holes,
                                                apply_local_elastic, apply_contour_smoothing,
                                                apply_local_brightness, apply_etch_bias,
                                                apply_local_affine, apply_shape_border,
                                                apply_corner_rounding)
from ..components.artifacts.geometric import apply_affine, apply_elastic
from ..components.artifacts.instrument_optical import (apply_psf_blur, apply_defocus_blur,
                                                     apply_charging, apply_topographic_shading,
                                                     apply_gradient_illumination, apply_striping_smearing,
                                                     apply_fixed_pattern_noise, apply_edge_brightness)
from ..outputs.writers import (save_numpy, save_image_data, save_json_data,
                             save_gif_data, save_text_file, calculate_hashes)
from ..outputs.ground_truth import (generate_instance_data, generate_combined_mask,
                                  generate_defect_mask, create_overlays,
                                  add_metadata_overlay)
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
    'corner_rounding': apply_corner_rounding
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
    
    # --- 1. Setup ---

    # Setup unique logger for this sample
    sample_logger = logging.getLogger(f"semgen_sample_{sample_idx:05d}")
    sample_logger.propagate = False
    sample_output_top_dir = Path(output_parent_dir) / f"sem_{sample_idx:05d}" # Dir for this sample's outputs
    log_dir = sample_output_top_dir / "logs"
    ensure_dir(log_dir)
    log_file_path = log_dir / "generation.log"
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file_path) for h in sample_logger.handlers):
        log_file_handler = logging.FileHandler(log_file_path, mode='w')
        log_file_handler.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s-%(name)s-%(filename)s:%(lineno)d- %(message)s'))
        sample_logger.addHandler(log_file_handler)
        sample_logger.setLevel(logging.DEBUG)

    start_time = time.time()
    sample_rng = get_rng(sample_seed)

    # --- Initialize ALL potential output variables to None outside the try block ---
    # This avoids UnboundLocalError if an error happens before assignment inside try
    image_final_noisy = None
    combined_mask_original_pre_warp_oversized = None
    semantic_map_warped = None
    layer_order_map_warped = None
    warp_field_final = None
    topography_height_map = None
    total_added_noise = None
    final_instance_mask = None
    final_instance_metadata = {}
    metadata = {}
    overlay_contour_vis, instance_mask_vis_overlay, warp_field_vis_overlay = None, None, None
    overlay_metadata_vis = None
    actual_layer_gif_frames = []
    output_paths = {}
    save_paths_list = []
    final_combined_actual_mask = None
    image_clean_pre_warp_oversized, image_post_geometric_warp, image_post_instrument = None, None, None
    # ---
    
    try:
        sample_config = base_config # Config is already randomized for this sample
        sample_logger.info(f"--- Generating Sample {sample_idx:05d} (Seed: {sample_seed}) ---")

        # --- Safely get image settings ---
        image_settings = sample_config.get('image_settings', {})
        final_h, final_w = image_settings.get('resolution', [256, 256]) # Final target dimensions
        
        # --- Calculate Oversized Dimensions UPFRONT ---
        margin_factor = image_settings.get('oversize_margin_factor', 0.25)
        margin_h_abs, margin_w_abs = int(final_h * margin_factor), int(final_w * margin_factor)
        current_h, current_w = final_h + 2 * margin_h_abs, final_w + 2 * margin_w_abs # WORKING dimensions
        oversized_shape_for_layers = (current_h, current_w)
        # ---

        bit_depth = image_settings.get('bit_depth', 16) # Default if missing
        magnification = image_settings.get('magnification', 1.0) # Default if missing
        pixel_size_nm_at_1x = image_settings.get('pixel_size_nm_at_1x', None) # Get safely
        pixel_size_nm = None
        if pixel_size_nm_at_1x is not None and magnification != 0:
            pixel_size_nm = pixel_size_nm_at_1x / magnification
        elif pixel_size_nm_at_1x is None:
            sample_logger.warning(f"Config missing 'pixel_size_nm_at_1x'.")
        elif magnification == 0: # Avoid division by zero
            sample_logger.warning(f"Magnification is zero, cannot calculate pixel size. Scale bar may be incorrect.")

        artifact_raffle_settings = sample_config.get('artifact_raffle', {})
        sample_logger.debug(f"Initializing Raffler with artifact_raffle_settings type: {type(artifact_raffle_settings)}")
        if isinstance(artifact_raffle_settings, dict):
            geo_artifacts_for_raffler = artifact_raffle_settings.get('categories',{}).get('geometric', 'MISSING_GEOMETRIC_CATEGORY')
            sample_logger.debug(f"Geometric artifacts passed to Raffler type: {type(geo_artifacts_for_raffler)}")
            if isinstance(geo_artifacts_for_raffler, list):
                names_for_raffler = [(item.get('name', 'NO_NAME'), item.get('probability', 'NO_PROB')) if isinstance(item, dict) else 'INVALID_ITEM_TYPE' for item in geo_artifacts_for_raffler]
                sample_logger.debug(f"Geometric artifact (name, prob) for Raffler: {names_for_raffler}")
                # sample_logger.debug(f"Full geometric artifact defs for Raffler: {json.dumps(geo_artifacts_for_raffler, indent=2)}") # More detail
            else:
                sample_logger.error(f"Geometric artifacts section for Raffler is not a list: {geo_artifacts_for_raffler}")
        else:
            sample_logger.error(f"Artifact raffle section for Raffler is not dict or missing: {artifact_raffle_settings}")

        geom_mode = artifact_raffle_settings.get('per_instance_geometric_mode', 'none') # Default to none
        out_opts = sample_config.get('output_options', {})
        run_settings = sample_config.get('run_settings', {}) # Get run_settings
        max_predictable_layers_for_gt = run_settings.get('max_predictable_layers', 5) # Get from config
        sample_logger.info(f"Image Size: {final_h}x{final_w}, Bit Depth: {bit_depth}, Magnification: {magnification:.2f}x")
        sample_logger.info(f"Per-instance Geometric Mode: {geom_mode}")

        # --- Initialize Raffler ---
        raff = Raffler(artifact_raffle_settings, sample_rng, sample_logger)
        
        # --- 2. Background ---
        # Generate background at WORKING (oversized) dimensions
        bg_conf = sample_config.get('background', {})
        background_clean = generate_background(bg_conf, (current_h, current_w), magnification, sample_rng, sample_logger)
        sample_logger.debug(f"Background generated. Is None: {background_clean is None}")
        if background_clean is not None:
            sample_logger.debug(f"Background shape: {background_clean.shape}, dtype: {background_clean.dtype}, sum: {background_clean.sum()}")

        initial_background = background_clean.copy()
        image_clean = background_clean.copy() # Composition starts here
        
        # --- 3. Layers ---
        layering_conf = sample_config.get('layering', {})
        selected_layers = layering_conf.get('selected_layers', [])
        composition_mode = layering_conf.get('composition_mode', 'additive')
        randomize_layer_order = layering_conf.get('randomize_order', False)
        layer_indices = list(range(len(selected_layers)))
        if randomize_layer_order: sample_rng.shuffle(layer_indices)
        
        # Ensure layer_colors has enough colors, avoid error if len(selected_layers) is 0
        num_unique_colors_needed = len(selected_layers) if selected_layers else 1
        layer_colors = get_distinct_colors(num_unique_colors_needed)
        layer_id_to_color = {i: layer_colors[i % len(layer_colors)] for i in range(len(selected_layers))}

        all_layers_data = {} # Stores data per layer (masks are oversized until saved)
        layer_masks_original = {} # Stores oversized original combined masks per layer
        layer_masks_actual = {}   # Stores oversized final actual masks (after layer etch bias)
        layer_renders_actual = [] # List of OVERSIZED final float renders for composition
        layer_renders_clean = []  # List of OVERSIZED clean renders
        instance_artifact_params_list = [] # Details of artifacts applied
        layer_defect_masks = {} # For oversized defect masks

        # Initialize global GT maps at OVERSIZED dimensions
        layer_order_map_pre_warp = np.zeros(oversized_shape_for_layers, dtype=np.uint8) if out_opts.get('save_layer_order_map') else None
        semantic_map_pre_warp = np.zeros(oversized_shape_for_layers, dtype=np.uint8) if out_opts.get('save_semantic_map') else None
        shape_type_semantic_mask_pre_warp = np.zeros(oversized_shape_for_layers, dtype=np.uint8) if out_opts.get('save_shape_type_semantic_mask', False) else None
        # For "X-Ray" vision: list to hold individual semantic masks for each layer
        per_layer_shape_type_semantic_masks_pre_warp = [] if out_opts.get('save_per_layer_shape_type_semantic_masks') else None        
        
        individual_layer_actual_masks_pre_warp = {} # Will store final oversized masks for global warping


        # --- 3a. Layer Generation Loop ---
        sample_logger.info(f"Processing {len(selected_layers)} selected layers on {current_w}x{current_h} canvas...")
        for layer_render_idx, layer_config_idx in enumerate(layer_indices):
            layer_conf = selected_layers[layer_config_idx]
            layer_id_str = f"layer_{layer_config_idx:02d}"
            layer_id_name = layer_conf.get('layer_id', layer_id_str)
            layer_output_dir = sample_output_top_dir / "layers" / layer_id_str # For CROPPED outputs
            ensure_dir(layer_output_dir)
            sample_logger.info(f" -> Layer {layer_render_idx+1}/{len(selected_layers)} (CfgIdx {layer_config_idx}): '{layer_id_name}'")

            intensity = layer_conf.get('intensity', 0.5)
            shape_type = layer_conf.get('shape', 'circle')
            alpha = layer_conf.get('alpha', 1.0)
            shape_params_base = layer_conf.get('shape_params', {})
            pattern_params = layer_conf.get('pattern_params', {})
            pattern_type = layer_conf.get('pattern', 'grid')

            current_layer_shape_type_str = layer_conf.get('shape', 'unknown_shape_type') # Default if missing
            current_shape_type_id = SHAPE_TYPE_MAP.get(current_layer_shape_type_str, SHAPE_TYPE_MAP["background"]) # Default to background ID
            if current_layer_shape_type_str == 'unknown_shape_type':
                sample_logger.warning(f"Layer {layer_id_name} has unknown shape type, using background ID for semantic mask.")
            layer_shape_artifacts_defs = raff.raffle_effects('shape')
            sample_logger.debug(f"    Raffled shape artifacts for layer: {[a.get('name', 'N/A') for a in layer_shape_artifacts_defs]}")

            brightness_artifact_def = next((a for a in layer_shape_artifacts_defs if a.get('name') == 'local_brightness'), None)
            layer_brightness_noise_map = None
            if brightness_artifact_def and HAS_PERLIN:
                try:
                    lb_params = brightness_artifact_def.get('params', {})
                    lb_scale = lb_params.get('scale', 20.0)
                    lb_octaves = lb_params.get('octaves', 4)
                    noise_gen = PerlinNoise(octaves=lb_octaves, seed=sample_rng.randint(0, 2**32 - 1))
                    layer_brightness_noise_map = np.zeros(oversized_shape_for_layers, dtype=np.float32)
                    for r_idx_noise in range(current_h):
                        for c_idx_noise in range(current_w):
                            layer_brightness_noise_map[r_idx_noise,c_idx_noise] = noise_gen([r_idx_noise/lb_scale, c_idx_noise/lb_scale])
                    mean_noise = np.mean(layer_brightness_noise_map); std_noise = np.std(layer_brightness_noise_map)
                    if std_noise > 1e-6: layer_brightness_noise_map = (layer_brightness_noise_map - mean_noise) / std_noise
                    else: layer_brightness_noise_map.fill(0.0)
                    sample_logger.debug("    Layer-wide brightness noise map generated.")
                except Exception as noise_err:
                    sample_logger.error(f"    Failed to generate layer brightness noise map: {noise_err}", exc_info=True)
                    layer_brightness_noise_map = None

            positions_or_paths = get_pattern_positions(pattern_type, oversized_shape_for_layers, shape_params_base, pattern_params, sample_rng, sample_logger)
            is_path_based = pattern_type in ['sine_wave_horizontal']
            sample_logger.debug(f"    Generated {len(positions_or_paths)} positions/paths for pattern '{pattern_type}'")

            layer_combined_mask_original_acc = np.zeros(oversized_shape_for_layers, dtype=np.uint8)
            layer_combined_mask_actual_acc = np.zeros(oversized_shape_for_layers, dtype=np.uint8)
            layer_render_buffer_final_acc = np.zeros(oversized_shape_for_layers, dtype=np.float32)
            layer_render_buffer_clean_acc = np.zeros(oversized_shape_for_layers, dtype=np.float32)
            # --- NEW: Semantic mask for THIS specific layer ---
            current_layer_semantic_mask_acc = np.zeros(oversized_shape_for_layers, dtype=np.uint8) \
                if per_layer_shape_type_semantic_masks_pre_warp is not None else None

            num_instances_in_layer = 0
            applied_shape_artifacts_names_layer = set()

            # --- 3b. Instance/Path Loop ---
            for idx, item in enumerate(positions_or_paths):
                shape_params = copy.deepcopy(shape_params_base)
                original_mask_instance, actual_mask_instance = None, None
                instance_render_clean, instance_render_final = None, None

                if is_path_based:
                    path_points = item
                    if not path_points or len(path_points) < 2: continue
                    thickness = max(1, int(round(shape_params.get('thickness', 2))))
                    temp_mask = np.zeros(oversized_shape_for_layers, dtype=np.uint8)
                    cv2.polylines(temp_mask, [np.array(path_points, dtype=np.int32)], isClosed=False, color=1, thickness=thickness)
                    original_mask_instance = temp_mask
                else: # Position-based
                    pos = item
                    if shape_type.endswith('line') and isinstance(pos, tuple) and len(pos) == 2 and isinstance(pos[0], tuple):
                         shape_params['x1'], shape_params['y1'] = pos[0]; shape_params['x2'], shape_params['y2'] = pos[1]
                    elif isinstance(pos, tuple) and len(pos) == 2 and isinstance(pos[0], (int, float)):
                         shape_params['center_x'], shape_params['center_y'] = pos
                    else: sample_logger.warning(f"    Unsupported pos for shape {shape_type}: {pos}"); continue
                    original_mask_instance = create_shape_mask(shape_type, shape_params, oversized_shape_for_layers, rng=sample_rng)

                if original_mask_instance is None or np.sum(original_mask_instance) == 0: continue
                layer_combined_mask_original_acc |= original_mask_instance

                actual_mask_instance = original_mask_instance.copy()
                for layer_artifact_def in layer_shape_artifacts_defs:
                    artifact_name = layer_artifact_def.get('name')
                    if not artifact_name or artifact_name in SHAPE_RENDER_ARTIFACT_FUNCS or artifact_name == 'etch_bias': continue
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
                    apply_this_artifact = True; target_mask_artifact_func = None
                    if artifact_name == 'local_elastic':
                        if geom_mode != 'local_elastic': apply_this_artifact = False
                        else: target_mask_artifact_func = SHAPE_MASK_ARTIFACT_FUNCS.get(artifact_name)
                    elif artifact_name == 'local_affine':
                        if geom_mode != 'local_affine': apply_this_artifact = False
                        else: target_mask_artifact_func = SHAPE_MASK_ARTIFACT_FUNCS.get(artifact_name)
                    elif artifact_name in SHAPE_MASK_ARTIFACT_FUNCS:
                        target_mask_artifact_func = SHAPE_MASK_ARTIFACT_FUNCS.get(artifact_name)
                    else: apply_this_artifact = False
                    if apply_this_artifact and target_mask_artifact_func:
                        try:
                            actual_mask_instance = target_mask_artifact_func(actual_mask_instance, instance_params, sample_rng, sample_logger)
                            applied_shape_artifacts_names_layer.add(artifact_name)
                            if out_opts.get('include_artifact_params_in_metadata', False):
                                instance_artifact_params_list.append({ 'sample_idx': sample_idx, 'instance_idx_in_layer': idx, 'layer_config_idx': layer_config_idx, 'artifact_name': artifact_name, 'parameters': copy.deepcopy(instance_params) })
                        except Exception as e: sample_logger.error(f"    Error applying mask artifact {artifact_name} to instance {idx}: {e}", exc_info=True)

                # --- Update Semantic Masks ---
                if actual_mask_instance is not None and np.sum(actual_mask_instance) > 0:
                    if shape_type_semantic_mask_pre_warp is not None and current_shape_type_id != SHAPE_TYPE_MAP["background"]:
                        shape_type_semantic_mask_pre_warp[actual_mask_instance > 0] = current_shape_type_id # Topmost wins

                    if current_layer_semantic_mask_acc is not None and current_shape_type_id != SHAPE_TYPE_MAP["background"]:
                        # For per-layer semantic mask, accumulate all shapes of this layer
                        current_layer_semantic_mask_acc[actual_mask_instance > 0] = current_shape_type_id
                # ---


                if shape_type_semantic_mask_pre_warp is not None and current_shape_type_id != SHAPE_TYPE_MAP["background"]:
                    # Topmost instance of a shape type wins for that pixel
                    shape_type_semantic_mask_pre_warp[actual_mask_instance > 0] = current_shape_type_id

                instance_render_clean = np.zeros_like(layer_render_buffer_final_acc)
                if is_path_based:
                    cv2.polylines(instance_render_clean, [np.array(item, dtype=np.int32)], isClosed=False, color=(intensity*alpha), thickness=max(1, int(round(shape_params.get('thickness', 2)))))
                    instance_render_clean[actual_mask_instance == 0] = 0
                else:
                    temp_render_geometry = render_shape(shape_type, shape_params, oversized_shape_for_layers, intensity=1.0, anti_aliasing=True, rng=sample_rng)
                    instance_render_clean[actual_mask_instance > 0] = temp_render_geometry[actual_mask_instance > 0] * intensity * alpha
                instance_render_clean = np.clip(instance_render_clean, 0.0, 1.0)
                layer_render_buffer_clean_acc += instance_render_clean

                instance_render_final = instance_render_clean.copy()
                render_artifact_order = ['shape_border', 'local_brightness']
                for artifact_name in render_artifact_order:
                    render_artifact_def = next((a for a in layer_shape_artifacts_defs if a.get('name') == artifact_name), None)
                    if render_artifact_def:
                        target_render_artifact_func = SHAPE_RENDER_ARTIFACT_FUNCS.get(artifact_name)
                        if target_render_artifact_func:
                            try:
                                layer_params_render = render_artifact_def.get('params', {})
                                instance_params_render = {}
                                variation_factor_render = 0.2
                                for p_name, p_val in layer_params_render.items():
                                    if isinstance(p_val, (int, float)):
                                        offset = p_val * variation_factor_render * sample_rng.uniform(-1.0, 1.0)
                                        if p_name == 'thickness': instance_params_render[p_name] = max(1, int(round(p_val + offset)))
                                        elif isinstance(p_val, int): instance_params_render[p_name] = max(0, int(round(p_val + offset)))
                                        else: instance_params_render[p_name] = p_val + offset
                                    else: instance_params_render[p_name] = p_val
                                if artifact_name == 'shape_border':
                                    instance_render_final = target_render_artifact_func(instance_render_final, actual_mask_instance, instance_params_render, intensity, sample_rng, sample_logger)
                                elif artifact_name == 'local_brightness':
                                    if layer_brightness_noise_map is not None:
                                        instance_contrast = instance_params_render.get('contrast', 0.1)
                                        masked_noise_region = actual_mask_instance > 0
                                        if np.any(masked_noise_region): # Ensure there are pixels to apply to
                                            noise_slice = layer_brightness_noise_map[masked_noise_region]
                                            brightness_variation = noise_slice * instance_contrast
                                            instance_render_final[masked_noise_region] *= (1.0 + brightness_variation)
                                            instance_render_final = np.clip(instance_render_final, 0.0, 1.0)
                                    else: sample_logger.warning("    Skipping local_brightness (no layer noise map).")
                                applied_shape_artifacts_names_layer.add(artifact_name)
                                if out_opts.get('include_artifact_params_in_metadata', False):
                                     instance_artifact_params_list.append({ 'sample_idx': sample_idx, 'instance_idx_in_layer': idx, 'layer_config_idx': layer_config_idx, 'artifact_name': artifact_name, 'parameters': copy.deepcopy(instance_params_render) })
                            except Exception as e: sample_logger.error(f"    Error applying render artifact {artifact_name} instance {idx}: {e}", exc_info=True)

                layer_combined_mask_actual_acc |= actual_mask_instance
                layer_render_buffer_final_acc += instance_render_final
                num_instances_in_layer += 1
            # --- End Instance/Path Loop ---

            layer_combined_mask_final_actual = layer_combined_mask_actual_acc
            etch_bias_def = next((a for a in layer_shape_artifacts_defs if a.get('name') == 'etch_bias'), None)
            if etch_bias_def:
                 try:
                    layer_etch_params = etch_bias_def.get('params', {})
                    layer_combined_mask_final_actual = apply_etch_bias(layer_combined_mask_final_actual, layer_etch_params, sample_rng, sample_logger)
                    applied_shape_artifacts_names_layer.add('etch_bias')
                    if out_opts.get('include_artifact_params_in_metadata', False):
                        instance_artifact_params_list.append({'sample_idx': sample_idx,'instance_idx_in_layer': -1, 'layer_config_idx': layer_config_idx, 'artifact_name': 'etch_bias', 'parameters': copy.deepcopy(layer_etch_params)})
                 except Exception as e_etch: sample_logger.error(f"    Error applying etch_bias to layer {layer_config_idx}: {e_etch}", exc_info=True)
            layer_masks_actual[layer_config_idx] = layer_combined_mask_final_actual # Store OVERSIZED final actual

            if layer_combined_mask_final_actual is not None: sample_logger.debug(f"    Layer {layer_config_idx} final actual mask (pre-warp) sum: {layer_combined_mask_final_actual.sum()}")
            individual_layer_actual_masks_pre_warp[layer_config_idx] = layer_combined_mask_final_actual.copy()

            if layer_order_map_pre_warp is not None: layer_order_map_pre_warp[layer_combined_mask_final_actual > 0] = layer_render_idx + 1
            # --- Add current layer's semantic mask to the list ---
            if per_layer_shape_type_semantic_masks_pre_warp is not None:
                if current_layer_semantic_mask_acc is not None:
                    per_layer_shape_type_semantic_masks_pre_warp.append(current_layer_semantic_mask_acc.copy())
                else: # Should not happen if main flag is true
                    per_layer_shape_type_semantic_masks_pre_warp.append(np.zeros(oversized_shape_for_layers, dtype=np.uint8))

            if semantic_map_pre_warp is not None: semantic_map_pre_warp[layer_combined_mask_final_actual > 0] = layer_config_idx + 1

            layer_render_buffer_final_acc = np.clip(layer_render_buffer_final_acc, 0.0, 1.0)
            layer_render_buffer_clean_acc = np.clip(layer_render_buffer_clean_acc, 0.0, 1.0)
            layer_renders_actual.append(layer_render_buffer_final_acc)
            layer_renders_clean.append(layer_render_buffer_clean_acc)

            sample_logger.debug(f"    Layer {layer_id_name}: Generated {num_instances_in_layer} instances.")
            defect_mask_layer = generate_defect_mask(layer_combined_mask_original_acc, layer_combined_mask_final_actual)
            all_layers_data[layer_config_idx] = {
                 'id': layer_id_name,
                 'original_mask_oversized': layer_combined_mask_original_acc.copy(),
                 'actual_mask_oversized': layer_combined_mask_final_actual.copy(),
                 'defect_mask_oversized': defect_mask_layer.copy() if defect_mask_layer is not None else None,
                 'render_clean_oversized': layer_render_buffer_clean_acc.copy() if out_opts.get('save_clean_layer_renders') else None,
                 'render_final_oversized': layer_render_buffer_final_acc.copy() if out_opts.get('save_per_layer_renders') else None,
                 'output_dir': layer_output_dir,
                 'num_instances': num_instances_in_layer,
                 'applied_shape_artifacts': sorted(list(applied_shape_artifacts_names_layer))
            }
            layer_masks_original[layer_config_idx] = layer_combined_mask_original_acc
            if defect_mask_layer is not None: layer_defect_masks[layer_config_idx] = defect_mask_layer

            # --- Save Per-Layer Outputs (CROPPED versions) ---
            crop_y_final_save = slice(margin_h_abs, margin_h_abs + final_h)
            crop_x_final_save = slice(margin_w_abs, margin_w_abs + final_w)
            if out_opts.get('save_masks'):
                 save_numpy(layer_combined_mask_original_acc[crop_y_final_save, crop_x_final_save], layer_output_dir / "original_mask.npy")
                 save_numpy(layer_combined_mask_final_actual[crop_y_final_save, crop_x_final_save], layer_output_dir / "actual_mask.npy")
                 if out_opts.get('save_defect_masks') and defect_mask_layer is not None:
                      save_numpy(defect_mask_layer[crop_y_final_save, crop_x_final_save], layer_output_dir / "defect_mask.npy")
            if out_opts.get('save_visualizations'):
                 save_image_data(layer_combined_mask_original_acc[crop_y_final_save, crop_x_final_save].astype(float), layer_output_dir / "original_mask_vis.png", 8)
                 save_image_data(layer_combined_mask_final_actual[crop_y_final_save, crop_x_final_save].astype(float), layer_output_dir / "actual_mask_vis.png", 8)
                 if out_opts.get('save_defect_masks') and defect_mask_layer is not None:
                      save_image_data(defect_mask_layer[crop_y_final_save, crop_x_final_save].astype(float), layer_output_dir / "defect_mask_vis.png", 8)
            if out_opts.get('save_clean_layer_renders') and all_layers_data[layer_config_idx]['render_clean_oversized'] is not None:
                save_image_data(all_layers_data[layer_config_idx]['render_clean_oversized'][crop_y_final_save, crop_x_final_save], layer_output_dir / "render_clean_vis.png", bit_depth)
            if out_opts.get('save_per_layer_renders') and all_layers_data[layer_config_idx]['render_final_oversized'] is not None:
                 save_image_data(all_layers_data[layer_config_idx]['render_final_oversized'][crop_y_final_save, crop_x_final_save], layer_output_dir / "render_final_vis.png", bit_depth)
        # --- End Layer Loop ---
        sample_logger.info("Finished processing all layers (on oversized canvas).")
        if per_layer_shape_type_semantic_masks_pre_warp is not None:
            num_generated_layers = len(per_layer_shape_type_semantic_masks_pre_warp)
            for _ in range(num_generated_layers, max_predictable_layers_for_gt):
                per_layer_shape_type_semantic_masks_pre_warp.append(np.zeros(oversized_shape_for_layers, dtype=np.uint8))
            # Truncate if more layers were generated than max_predictable (less common)
            per_layer_shape_type_semantic_masks_pre_warp = per_layer_shape_type_semantic_masks_pre_warp[:max_predictable_layers_for_gt]

        
        # --- 4. Compose Layers ---
        # image_clean is already oversized and started with background
        # layer_renders_actual contains oversized renders
        sample_logger.info(f"Composing {len(layer_renders_actual)} layers using mode: {composition_mode}")
        cumulative_layers_for_gif_oversized = [initial_background.copy()] if initial_background is not None else []

        for i_comp, layer_buffer_oversized in enumerate(layer_renders_actual):
             sample_logger.debug(f"  Composing layer {i_comp}. image_clean is None: {image_clean is None}. layer_buffer_oversized is None: {layer_buffer_oversized is None}")
             if layer_buffer_oversized is not None:
                 sample_logger.debug(f"    Layer buffer sum: {layer_buffer_oversized.sum()}")

             if image_clean is None and layer_buffer_oversized is not None:
                 sample_logger.warning("image_clean was None, initializing with current layer_buffer_oversized.")
                 image_clean = layer_buffer_oversized.copy()
             elif image_clean is not None and layer_buffer_oversized is not None:             
                 if composition_mode == 'additive': image_clean += layer_buffer_oversized
                 elif composition_mode == 'multiplicative': image_clean *= (1.0 + layer_buffer_oversized * 0.5) # Softer mult
                 elif composition_mode == 'overwrite':
                       # Determine original index based on potentially shuffled layer_indices
                       original_layer_idx_for_mask = layer_indices[i_comp]
                       mask_oversized = layer_masks_actual.get(original_layer_idx_for_mask)
                       if mask_oversized is not None: image_clean[mask_oversized > 0] = layer_buffer_oversized[mask_oversized > 0]
                       else: sample_logger.warning(f"Mask not found for layer (cfg_idx {original_layer_idx_for_mask}) during overwrite composition.")
                 else: image_clean += layer_buffer_oversized # Default additive
                 image_clean = np.clip(image_clean, 0.0, 1.0)
             if out_opts.get('save_gifs'): # Only append if saving gifs
                  cumulative_layers_for_gif_oversized.append(image_clean.copy())

        sample_logger.debug(f"After composition, image_clean is None: {image_clean is None}")
        image_clean_pre_warp_oversized = image_clean.copy() if image_clean is not None else None
        sample_logger.debug(f"After composition, image_clean_pre_warp_oversized is None: {image_clean_pre_warp_oversized is None}")
        if image_clean_pre_warp_oversized is not None:
            sample_logger.debug(f"  Shape: {image_clean_pre_warp_oversized.shape}, Sum: {image_clean_pre_warp_oversized.sum()}")

        sample_logger.debug("Layer composition complete (on oversized canvas).")
        
        # --- 5. Generate Combined Original Mask (Pre-Warp) ---
        combined_mask_original_pre_warp_oversized = generate_combined_mask(layer_masks_original)

        # --- 6. Global Geometric Warping ---
        sample_logger.info("Applying global geometric artifacts to oversized data...")

        # These inputs are already at oversized dimensions from previous steps
        image_to_warp = image_clean_pre_warp_oversized
        current_semantic_map_to_warp = semantic_map_pre_warp
        current_layer_order_map_to_warp = layer_order_map_pre_warp

        # Prepare the list of all mask-like arrays that need to be warped together
        # This list maintains a specific order for later extraction.
        all_maps_and_masks_for_warping = []
        num_general_maps_warped = 0 # Count how many non-layer-specific maps are added

        if shape_type_semantic_mask_pre_warp is not None:
            all_maps_and_masks_for_warping.append(shape_type_semantic_mask_pre_warp)
        if current_semantic_map_to_warp is not None:
            all_maps_and_masks_for_warping.append(current_semantic_map_to_warp)
            num_general_maps_warped += 1
        if current_layer_order_map_to_warp is not None:
            all_maps_and_masks_for_warping.append(current_layer_order_map_to_warp)
            num_general_maps_warped += 1

        # Add per-layer semantic masks (if generating them)
        num_per_layer_semantic_masks_warped = 0
        if per_layer_shape_type_semantic_masks_pre_warp is not None:
            for pl_sem_mask in per_layer_shape_type_semantic_masks_pre_warp: # Should be padded to max_predictable_layers
                all_maps_and_masks_for_warping.append(pl_sem_mask)
            num_per_layer_semantic_masks_warped = len(per_layer_shape_type_semantic_masks_pre_warp)


        # Add individual layer actual masks (these are final post-layer-artifacts, oversized)
        # Ensure a consistent order for adding and later extracting these
        sorted_layer_indices_for_individual_masks = sorted(individual_layer_actual_masks_pre_warp.keys())
        for layer_idx in sorted_layer_indices_for_individual_masks:
            mask = individual_layer_actual_masks_pre_warp.get(layer_idx)
            if mask is not None:
                all_maps_and_masks_for_warping.append(mask)
            else: # Should not happen if populated correctly, but good to handle
                sample_logger.warning(f"Missing oversized mask for layer index {layer_idx} before warping. Appending None.")
                all_maps_and_masks_for_warping.append(None) # Maintain list length

        geometric_artifacts = raff.raffle_effects('geometric')
        warp_field_combined_oversized = None # Will store the final warp field for the image
        applied_geometric_artifacts = []

        # Sequentially apply each raffled geometric artifact
        for artifact_def in geometric_artifacts:
             func = GEOMETRIC_ARTIFACT_FUNCS.get(artifact_def['name'])
             if func:
                  try:
                       sample_logger.info(f"Applying geometric artifact: {artifact_def['name']}")
                       # The func is expected to take the image and the LIST of masks,
                       # and return the warped image, a LIST of warped masks, and the warp field
                       image_to_warp, all_maps_and_masks_for_warping, current_warp_field_oversized = \
                           func(image_to_warp, all_maps_and_masks_for_warping, artifact_def['params'], sample_rng, sample_logger)

                       if current_warp_field_oversized is not None:
                            warp_field_combined_oversized = current_warp_field_oversized # Store last one
                       applied_geometric_artifacts.append(artifact_def['name'])
                  except Exception as e_geo_art:
                       sample_logger.error(f"Error applying geometric artifact '{artifact_def['name']}': {e_geo_art}", exc_info=True)
             else:
                  sample_logger.warning(f"Geometric artifact function '{artifact_def['name']}' not found.")

        # --- Center Crop ALL warped items back to FINAL target resolution ---
        sample_logger.info(f"Cropping warped data from {current_w}x{current_h} to {final_w}x{final_h}")
        # --- Define and Save `image_clean_pre_warp.png` (CROPPED version) ---
        image_clean_pre_warp_final_res = None
        if image_clean_pre_warp_oversized is not None:
            crop_y_final = slice(margin_h_abs, margin_h_abs + final_h)
            crop_x_final = slice(margin_w_abs, margin_w_abs + final_w)
            image_clean_pre_warp_final_res = image_clean_pre_warp_oversized[crop_y_final, crop_x_final]
            sample_logger.debug(f"Created image_clean_pre_warp_final_res (cropped). Shape: {getattr(image_clean_pre_warp_final_res, 'shape', 'None')}")
        else:
            sample_logger.warning("image_clean_pre_warp_oversized was None. Cannot create cropped version for saving.")


        sample_logger.debug(f"Before geometric artifact loop, image_to_warp is None: {image_to_warp is None}")
        if image_to_warp is not None:
            sample_logger.debug(f"image_to_warp shape: {image_to_warp.shape}, dtype: {image_to_warp.dtype}")

        image_clean_warped = image_to_warp[crop_y_final, crop_x_final] if image_to_warp is not None else None
        warp_field_final = warp_field_combined_oversized[crop_y_final, crop_x_final] if warp_field_combined_oversized is not None else None

        # Extract and crop warped general maps from the `all_maps_and_masks_for_warping` list
        processed_mask_idx = 0 # Keep track of which mask we are extracting
        shape_type_semantic_mask_warped = None # Initialize
        if shape_type_semantic_mask_pre_warp is not None: # Check if it was part of warping
            if len(all_maps_and_masks_for_warping) > processed_mask_idx and all_maps_and_masks_for_warping[processed_mask_idx] is not None:
                shape_type_semantic_mask_warped = all_maps_and_masks_for_warping[processed_mask_idx][crop_y_final, crop_x_final]
            processed_mask_idx += 1

        semantic_map_warped = None
        if current_semantic_map_to_warp is not None: # Check if it was part of warping
            if len(all_maps_and_masks_for_warping) > processed_mask_idx and all_maps_and_masks_for_warping[processed_mask_idx] is not None:
                semantic_map_warped = all_maps_and_masks_for_warping[processed_mask_idx][crop_y_final, crop_x_final]
            processed_mask_idx += 1
        layer_order_map_warped = None
        if current_layer_order_map_to_warp is not None: # Check if it was part of warping
            if len(all_maps_and_masks_for_warping) > processed_mask_idx and all_maps_and_masks_for_warping[processed_mask_idx] is not None:
                layer_order_map_warped = all_maps_and_masks_for_warping[processed_mask_idx][crop_y_final, crop_x_final]
            processed_mask_idx += 1

        # Extract per-layer semantic masks
        warped_per_layer_shape_type_semantic_masks = []
        if per_layer_shape_type_semantic_masks_pre_warp is not None:
            for i in range(num_per_layer_semantic_masks_warped):
                if len(all_maps_and_masks_for_warping) > processed_mask_idx and all_maps_and_masks_for_warping[processed_mask_idx] is not None:
                    warped_per_layer_shape_type_semantic_masks.append(all_maps_and_masks_for_warping[processed_mask_idx][crop_y_final, crop_x_final])
                else: # Append an empty mask if something went wrong
                    warped_per_layer_shape_type_semantic_masks.append(np.zeros((final_h, final_w), dtype=np.uint8))
                processed_mask_idx += 1


        # Extract and crop warped individual layer masks
        warped_individual_layer_masks = {} # Stores CROPPED warped individual layer masks
        for layer_idx in sorted_layer_indices_for_individual_masks:
            # Only try to extract if it was originally present
            if individual_layer_actual_masks_pre_warp.get(layer_idx) is not None:
                if len(all_maps_and_masks_for_warping) > processed_mask_idx and all_maps_and_masks_for_warping[processed_mask_idx] is not None:
                    warped_mask_full = all_maps_and_masks_for_warping[processed_mask_idx]
                    warped_individual_layer_masks[layer_idx] = warped_mask_full[crop_y_final, crop_x_final]
                    if out_opts.get('save_warped_layer_masks', False): # Optional save
                         if layer_idx in all_layers_data:
                            all_layers_data[layer_idx]['actual_mask_warped'] = warped_individual_layer_masks[layer_idx]
                else:
                    sample_logger.warning(f"Expected warped mask for layer index {layer_idx} not found or list too short at index {processed_mask_idx}.")
                processed_mask_idx += 1 # Increment for each potential individual layer mask
        # --- End Cropping ---

        image_post_geometric_warp = image_clean_warped.copy() if image_clean_warped is not None else None
        sample_logger.debug(f"image_post_geometric_warp is None: {image_post_geometric_warp is None}")

        def add_path(key, path_obj):
            if path_obj and path_obj.exists():
                try:
                    output_paths[key] = str(path_obj.relative_to(output_parent_dir))
                    save_paths_list.append(path_obj)
                except ValueError: output_paths[key] = str(path_obj); save_paths_list.append(path_obj)
            elif path_obj: output_paths[key] = f"MISSING: {path_obj.name}"        

        # --- 7. Save Post-Geometric Intermediate ---
        if out_opts.get('save_extra_intermediate', False):
            path_post_geo = sample_output_top_dir / "image_post_geometric_warp.png"
            sample_logger.debug(f"Attempting to save image_post_geometric_warp to {path_post_geo}. Is None: {image_post_geometric_warp is None}")
            save_image_data(image_post_geometric_warp, path_post_geo, 8)
            add_path('image_post_geometric_warp', path_post_geo) # Helper to add to a temp list for metadata

        # --- 8. Apply Instrument Artifacts ---
        sample_logger.info("Applying instrument artifacts...")
        image_post_instrument = image_post_geometric_warp
        applied_instrument_artifacts = []
        total_added_fpn = np.zeros_like(image_post_instrument) if image_post_instrument is not None else None
        topography_height_map = None # Initialize

        for artifact in raff.raffle_effects('instrument'): # Raffle instrument effects
            func = INSTRUMENT_ARTIFACT_FUNCS.get(artifact['name'])
            if func:
                try:
                    sample_logger.info(f"Applying instrument artifact: {artifact['name']}")
                    params = artifact.get('params', {})
                    if artifact['name'] == 'topographic_shading':
                        image_post_instrument, height_map_generated = func(image_post_instrument, all_layers_data, params, sample_rng, sample_logger)
                        if out_opts.get('save_topography_height_map', False): topography_height_map = height_map_generated
                    elif artifact['name'] == 'fixed_pattern_noise':
                        image_post_instrument, fpn_map = func(image_post_instrument, params, sample_rng, sample_logger)
                        if total_added_fpn is not None and fpn_map is not None: total_added_fpn += fpn_map
                    else:
                        image_post_instrument = func(image_post_instrument, params, sample_rng, sample_logger)
                    applied_instrument_artifacts.append(artifact['name'])
                except Exception as e_inst_art: sample_logger.error(f"Error applying instrument artifact {artifact['name']}: {e_inst_art}", exc_info=True)
            else: sample_logger.warning(f"Instrument artifact function '{artifact['name']}' not found.")

        if image_post_instrument is not None: image_post_instrument = np.clip(image_post_instrument, 0.0, 1.0)
        sample_logger.debug(f"image_post_instrument is None: {image_post_instrument is None}")

        # --- 9. Save Post-Instrument Intermediate ---
        if out_opts.get('save_extra_intermediate', False) and image_post_instrument is not None:
            path_post_inst = sample_output_top_dir / "image_post_instrument.png"
            save_image_data(image_post_instrument, path_post_inst, 8)
            add_path('image_post_instrument', path_post_inst)

        # --- 10. Apply Detector Noise ---
        sample_logger.info("Applying detector noise...")
        image_final_noisy = image_post_instrument.copy() if image_post_instrument is not None else None
        if image_final_noisy is not None :
            total_added_noise = total_added_fpn.copy() if total_added_fpn is not None else np.zeros_like(image_final_noisy)
            applied_noise_artifacts = []
            noise_artifacts = raff.raffle_effects('noise')

            # Apply quantization first if present
            quant_artifact = next((a for a in noise_artifacts if a.get('name') == 'quantization'), None)
            if quant_artifact:
                try:
                    logger.info(f"Applying noise: {quant_artifact['name']}")
                    image_final_noisy, q_noise = apply_noise(image_final_noisy, quant_artifact['name'], quant_artifact['params'], sample_rng, sample_logger)
                    total_added_noise += q_noise
                    applied_noise_artifacts.append(quant_artifact['name'])
                except Exception as e: sample_logger.error(f"Error applying quantization: {e}", exc_info=True)
            # Apply other noise types
            for artifact in noise_artifacts:
                if artifact.get('name') == 'quantization': 
                    continue
                noise_type = artifact.get('name')
                if not noise_type:
                    continue
                try:
                    sample_logger.info(f"Applying noise: {noise_type}")
                    image_final_noisy, added_noise = apply_noise(image_final_noisy, noise_type, artifact['params'], sample_rng, sample_logger)
                    if added_noise is not None:
                        total_added_noise += added_noise
                    applied_noise_artifacts.append(noise_type)
                except Exception as e_noise:
                    sample_logger.error(f"Error applying noise {noise_type}: {e_noise}", exc_info=True)
            image_final_noisy = np.clip(image_final_noisy, 0.0, 1.0)
        sample_logger.debug(f"image_final_noisy is None: {image_final_noisy is None} (before final save attempt)")
        sample_logger.debug("Detector noise complete.")

        # --- 11. Generate Final Instance GT using WARPED individual layer masks ---
        sample_logger.info("Generating final instance data from warped layer masks...")
        final_instance_mask = None
        final_instance_metadata = {}

        # Ensure the flag exists and is true, and we have the necessary masks
        sample_logger.debug(f"Flag 'save_bounding_boxes': {out_opts.get('save_bounding_boxes', False)}")
        if warped_individual_layer_masks:
            sample_logger.debug(f"Number of warped individual layer masks: {len(warped_individual_layer_masks)}")
            for l_idx, l_mask in warped_individual_layer_masks.items():
                if l_mask is not None:
                    sample_logger.debug(f" Layer {l_idx} mask - Shape: {l_mask.shape}, Sum: {l_mask.sum()}, Dtype: {l_mask.dtype}")
                else:
                    sample_logger.debug(f" Layer {l_idx} mask is None")
        else:
            sample_logger.debug("warped_individual_layer_masks is empty or None.")

        if out_opts.get('save_bounding_boxes', False):
            sample_logger.debug(f"Attempting to generate instance data because 'save_bounding_boxes' is True.")
            # Log status of inputs
            sample_logger.debug(f"warped_individual_layer_masks is present: {bool(warped_individual_layer_masks)}")
            if warped_individual_layer_masks:
                 for l_idx, l_mask_val in warped_individual_layer_masks.items():
                      is_valid_mask = l_mask_val is not None and isinstance(l_mask_val, np.ndarray) and l_mask_val.sum() > 0
                      sample_logger.debug(f"  Layer {l_idx} in warped_individual_layer_masks. Valid for GT: {is_valid_mask}")

            sample_logger.debug(f"semantic_map_warped is present: {semantic_map_warped is not None}")
            if semantic_map_warped is not None:
                 sample_logger.debug(f"semantic_map_warped sum: {semantic_map_warped.sum()}")

            # Call generate_instance_data, it will decide which input to use
            try:
                final_instance_mask, final_instance_metadata = generate_instance_data(
                    layer_masks_actual=warped_individual_layer_masks, # Pass dict, can be empty
                    warped_semantic_map=semantic_map_warped      # Pass map, can be None
                )
                if final_instance_mask is not None and final_instance_metadata:
                    sample_logger.info(f"Generated {len(final_instance_metadata)} instance annotations.")
                elif not final_instance_metadata: # If it returned empty metadata
                    sample_logger.warning("generate_instance_data returned empty metadata (no instances found).")
                    # Keep final_instance_mask as None if no instances were truly found
                    if final_instance_mask is not None and final_instance_mask.sum() == 0:
                        final_instance_mask = None

            except Exception as e_inst:
                 sample_logger.error(f"Error calling generate_instance_data: {e_inst}", exc_info=True)
                 final_instance_metadata = {'error': "Failed: generate_instance_data call."}
        else:
            sample_logger.info("Skipping instance data generation ('save_bounding_boxes' is False).")


        # --- Generate Final Combined Actual Mask from CROPPED warped individual layer masks ---
        final_combined_actual_mask = None
        if out_opts.get('save_masks', False): # Check if we even need to generate it
            if warped_individual_layer_masks and any(m is not None and m.sum() > 0 for m in warped_individual_layer_masks.values()):
                final_combined_actual_mask = generate_combined_mask(warped_individual_layer_masks)
                sample_logger.debug(f"Generated final_combined_actual_mask sum: {final_combined_actual_mask.sum() if final_combined_actual_mask is not None else 'None'}")
            elif final_instance_mask is not None and final_instance_mask.sum() > 0 : # Fallback
                final_combined_actual_mask = (final_instance_mask > 0).astype(np.uint8)
                sample_logger.debug(f"Generated final_combined_actual_mask from final_instance_mask, sum: {final_combined_actual_mask.sum()}")
            else:
                sample_logger.warning("Could not generate final_combined_actual_mask (no valid input).")


        # --- 12. Generate Overlays ---
        sample_logger.info("Generating overlays...")
        final_image_vis_8bit = image_to_bit_depth(image_final_noisy, 8) if image_final_noisy is not None else None
        overlay_contour_vis, instance_mask_vis_overlay, warp_field_vis_overlay = None, None, None # Init with correct variable name
        if final_image_vis_8bit is not None:
            try:
                sample_logger.debug(f"Creating overlays with final_combined_actual_mask sum: {final_combined_actual_mask.sum() if final_combined_actual_mask is not None else 'None'}")
                sample_logger.debug(f"Creating overlays with final_instance_mask sum: {final_instance_mask.sum() if final_instance_mask is not None else 'None'}")

                overlay_contour_vis, instance_mask_vis_overlay, warp_field_vis_overlay = create_overlays(
                    final_image_vis_8bit,
                    final_combined_actual_mask, # Use THE final combined mask
                    final_instance_mask,        # Use THE final instance mask
                    warp_field_final,
                    layer_id_to_color
                )
            except Exception as e_overlay: sample_logger.error(f"Error generating overlays: {e_overlay}", exc_info=True)
            metadata_text = f"Sample: {sample_idx:05d}\nSeed: {sample_seed}\nMag: {magnification:.2f}x"
            overlay_metadata_vis = add_metadata_overlay(final_image_vis_8bit, text=metadata_text, pixel_size_nm=pixel_size_nm)

        if out_opts.get('save_gifs') and cumulative_layers_for_gif_oversized:
            for frame in cumulative_layers_for_gif_oversized:
                actual_layer_gif_frames.append(image_to_bit_depth(frame, 8))
    
        # --- 13. Prepare Metadata ---
        metadata = {
            "sample_index": sample_idx,
            "sample_name": f"sem_{sample_idx:05d}",
            "seed": sample_seed,
            "resolution": [final_h, final_w],
            "magnification": magnification,
            "pixel_size_nm_at_1x": pixel_size_nm_at_1x,
            "pixel_size_nm_calculated": pixel_size_nm,
            "background_type": bg_conf.get('types'), # Use resolved type
            "layers_used": [{'config_idx': idx, 'layer_id': data['id']} for idx, data in all_layers_data.items()],
            "num_layers_generated": len(selected_layers),
            "composition_mode": composition_mode,
            "applied_shape_artifacts": list(set(a for data in all_layers_data.values() for a in data['applied_shape_artifacts'])), # Unique names across layers
            "applied_geometric_artifacts": applied_geometric_artifacts,
            "applied_instrument_artifacts": applied_instrument_artifacts,
            "applied_noise_artifacts": applied_noise_artifacts,
            "instance_annotations": final_instance_metadata if out_opts.get('save_bounding_boxes', False) else None,
            "instance_artifact_details": instance_artifact_params_list if out_opts.get('include_artifact_params_in_metadata', False) else None,
            "generation_time_sec": round(time.time() - start_time, 2),
            "output_paths": {} # Filled during saving
        }
        # Remove empty keys if needed
        if not metadata.get("instance_annotations"): metadata.pop("instance_annotations", None)
        if not metadata.get("instance_artifact_details"): metadata.pop("instance_artifact_details", None)

        # --- 14. Save Final Outputs ---
        sample_logger.info("Saving final outputs...")

        # --- Main Image ---
        final_image_format = out_opts.get('output_formats', {}).get('final_image', 'tif').lower()
        final_img_path = sample_output_top_dir.parent / f"{sample_output_top_dir.name}.{final_image_format}" # Save next to folder
        
        # --- *** Add Debugging Here *** ---
        if image_final_noisy is not None:
            sample_logger.debug(f"Preparing to save final image to: {final_img_path}")
            sample_logger.debug(f"Data type: {image_final_noisy.dtype}, Shape: {image_final_noisy.shape}")
            try:
                min_val, max_val = np.min(image_final_noisy), np.max(image_final_noisy)
                sample_logger.debug(f"Data range: min={min_val:.4f}, max={max_val:.4f}")
                is_finite = np.all(np.isfinite(image_final_noisy))
                sample_logger.debug(f"All finite: {is_finite}")
                if not is_finite:
                     sample_logger.warning("Non-finite values (NaN/inf) found in final image data!")
                     # Optionally clip or replace NaNs here if needed before saving
                     # image_final_noisy = np.nan_to_num(image_final_noisy, nan=0.0, posinf=1.0, neginf=0.0)
            except Exception as e_check:
                 sample_logger.error(f"Error checking final image data: {e_check}", exc_info=True)
        else:
            sample_logger.error("!!! Cannot save final image: 'image_final_noisy' is None !!!")
        # --- *** End Debugging *** ---

        
        save_success = save_image_data(image_final_noisy, final_img_path, bit_depth, format_hint=final_image_format.upper())
        if save_success:
             sample_logger.debug(f"Successfully saved final image: {final_img_path}")
             add_path('final_image', final_img_path)
        else:
             sample_logger.error(f"!!! Failed to save final image file at: {final_img_path} (see writer logs) !!!")
             add_path('final_image', None) # Indicate missing file in metadata

        # --- Intermediate Images ---
        if out_opts.get('save_intermediate', False):
             path = sample_output_top_dir / "image_clean_pre_warp.png"
             sample_logger.debug(f"Attempting to save image_clean_pre_warp_final_res to {path}. Is None: {image_clean_pre_warp_final_res is None}")             
             save_success = save_image_data(image_clean_pre_warp_final_res, path, 8) # Save the CROPPED version
             if save_success: add_path('image_clean_pre_warp', path)
             else: sample_logger.error(f"Failed to save image_clean_pre_warp.png")
        if out_opts.get('save_extra_intermediate', False):
             path_post_geo = sample_output_top_dir / "image_post_geometric_warp.png"
             save_image_data(image_post_geometric_warp, path_post_geo, 8)
             add_path('image_post_geometric_warp', path_post_geo)
             path_post_inst = sample_output_top_dir / "image_post_instrument.png"
             save_image_data(image_post_instrument, path_post_inst, 8)
             add_path('image_post_instrument', path_post_inst)
        path_final_vis = sample_output_top_dir / "image_final_noisy_vis.png"
        save_image_data(image_final_noisy, path_final_vis, 8)
        add_path('image_final_noisy_vis', path_final_vis)
        # Background
        path_bg_npy = sample_output_top_dir / "layers_combined" / "background.npy"
        save_numpy(initial_background, path_bg_npy)
        add_path('background_npy', path_bg_npy)
        path_bg_vis = sample_output_top_dir / "layers_combined" / "background_vis.png"
        save_image_data(initial_background, path_bg_vis, 8)
        add_path('background_vis', path_bg_vis)


        # --- Masks and GT Maps ---
        if out_opts.get('save_masks'):
            # Combined Original (Pre-Warp) - This one IS intentionally unwarped
            if combined_mask_original_pre_warp_oversized is not None:
                path_com = sample_output_top_dir / "combined_original_mask.npy"
                save_numpy(combined_mask_original_pre_warp_oversized, path_com)
                add_path('combined_original_mask_npy', path_com)
                if out_opts.get('save_visualizations'):
                    path_co_vis = sample_output_top_dir / "combined_original_mask_vis.png"
                    save_image_data(combined_mask_original_pre_warp_oversized.astype(float), path_co_vis, 8)
                    add_path('combined_original_mask_vis', path_co_vis)

            # --- Save FINAL Combined Actual Mask (WARPED) ---
            if final_combined_actual_mask is not None: # This is the WARPED version
                path_ca = sample_output_top_dir / "combined_actual_mask.npy"
                save_numpy(final_combined_actual_mask, path_ca)
                add_path('combined_actual_mask_npy', path_ca)
                if out_opts.get('save_visualizations'):
                    path_ca_vis = sample_output_top_dir / "combined_actual_mask_vis.png"
                    sample_logger.debug(f"Saving combined_actual_mask_vis.png with sum: {final_combined_actual_mask.sum()}")
                    save_image_data(final_combined_actual_mask.astype(float), path_ca_vis, 8) # Use float for vis
                    add_path('combined_actual_mask_vis', path_ca_vis)
            else:
                sample_logger.warning("final_combined_actual_mask is None, not saving.")

             # Final Instance Mask (WARPED)
            if final_instance_mask is not None: # This is the WARPED version
                inst_mask_format = out_opts.get('output_formats', {}).get('instance_mask', 'tif').lower()
                path_inst = sample_output_top_dir / f"instance_mask.{inst_mask_format}"
                # ... (logic to save final_instance_mask, fallback to NPY if image format fails - as before) ...
                # Ensure the 'save_image_data' call uses final_instance_mask
                # Example (simplified, actual logic is more complex with fallback):
                save_image_data(final_instance_mask, path_inst, bit_depth=32 if final_instance_mask.dtype==np.uint32 else 16, format_hint=inst_mask_format.upper())
                add_path('instance_mask', path_inst)
                # The visualization (instance_mask_vis.png) is saved in the "Overlays" section using 'instance_mask_vis_overlay'
            else:
                sample_logger.warning("final_instance_mask is None, not saving main instance file.")

        # --- Save Per-Layer Shape Type Semantic Masks ---
        if out_opts.get('save_per_layer_shape_type_semantic_masks', False) and warped_per_layer_shape_type_semantic_masks:
            for i, layer_sem_mask_warped in enumerate(warped_per_layer_shape_type_semantic_masks):
                # Only save up to the actual number of layers generated for this sample,
                # or up to max_predictable_layers if that's how the list was padded
                # Let's use number of selected_layers
                if i < len(selected_layers): # Match original layer index
                    # Find original layer_config_idx based on render order if layers were shuffled
                    original_cfg_idx = layer_indices[i] # Use the i-th element from the (possibly shuffled) render order
                    layer_id_for_filename = all_layers_data.get(original_cfg_idx, {}).get('id', f'layer_{original_cfg_idx:02d}')

                    path_pls = sample_output_top_dir / "layers" / f"layer_{original_cfg_idx:02d}" / f"shape_type_semantic_mask.npy"
                    ensure_dir(path_pls.parent) # Ensure layer_XX dir exists
                    save_numpy(layer_sem_mask_warped, path_pls)
                    add_path(f'layer_{original_cfg_idx:02d}_shape_type_semantic_mask_npy', path_pls)

                    if out_opts.get('save_visualizations'):
                        shape_vis_colors = get_distinct_colors(NUM_SHAPE_CLASSES)
                        shape_type_colormap = {k: shape_vis_colors[k % len(shape_vis_colors)] for k in range(NUM_SHAPE_CLASSES)}
                        pls_vis = create_color_visualization(layer_sem_mask_warped, shape_type_colormap)
                        save_image_data(pls_vis, path_pls.with_suffix(".png"), 8)
                        add_path(f'layer_{original_cfg_idx:02d}_shape_type_semantic_mask_vis', path_pls.with_suffix(".png"))
                elif i < max_predictable_layers_for_gt: # Save empty masks for padding if needed by training
                    path_pls_empty = sample_output_top_dir / "layers" / f"layer_padding_{i:02d}" / f"shape_type_semantic_mask.npy"
                    ensure_dir(path_pls_empty.parent)
                    save_numpy(layer_sem_mask_warped, path_pls_empty) # Save the zero mask
                    # Don't necessarily add these padding masks to metadata output_paths unless useful



        # --- Save SHAPE TYPE Semantic Mask ---
        if out_opts.get('save_shape_type_semantic_mask', False) and shape_type_semantic_mask_warped is not None:
            path_sts = sample_output_top_dir / "shape_type_semantic_mask.npy"
            save_numpy(shape_type_semantic_mask_warped, path_sts) # Already uint8
            add_path('shape_type_semantic_mask_npy', path_sts)
            if out_opts.get('save_visualizations'):
                shape_vis_colors = get_distinct_colors(NUM_SHAPE_CLASSES)
                shape_type_colormap = {i: shape_vis_colors[i % len(shape_vis_colors)] for i in range(NUM_SHAPE_CLASSES)}
                sts_vis = create_color_visualization(shape_type_semantic_mask_warped, shape_type_colormap)
                save_image_data(sts_vis, sample_output_top_dir / "shape_type_semantic_mask_vis.png", 8)
                add_path('shape_type_semantic_mask_vis', sample_output_top_dir / "shape_type_semantic_mask_vis.png")


        # Layer Order Map
        if out_opts.get('save_layer_order_map', False) and layer_order_map_warped is not None:
            path_lom = sample_output_top_dir / "layer_order_map.npy"
            save_numpy(layer_order_map_warped.astype(np.uint8), path_lom)
            add_path('layer_order_map_npy', path_lom)
            if out_opts.get('save_visualizations'):
                lom_vis_gray = normalize_image(layer_order_map_warped)
                path_lom_vis = sample_output_top_dir / "layer_order_map_vis.png"
                save_image_data(lom_vis_gray, path_lom_vis, 8)
                add_path('layer_order_map_vis', path_lom_vis)

        # Semantic Map
        if semantic_map_warped is not None:
            sample_logger.debug(f"Warped semantic map - Shape: {semantic_map_warped.shape}, Sum: {semantic_map_warped.sum()}, Dtype: {semantic_map_warped.dtype}")
        else:
            sample_logger.debug("semantic_map_warped is None.")
        if out_opts.get('save_semantic_map', False) and semantic_map_warped is not None:
             path_sem = sample_output_top_dir / "semantic_map_layer_idx.npy"
             save_numpy(semantic_map_warped.astype(np.uint8), path_sem)
             add_path('semantic_map_npy', path_sem)
             if out_opts.get('save_visualizations'):
                max_layer_idx = np.max(semantic_map_warped)
                sem_colormap = {i: layer_id_to_color.get(i-1, (128,128,128)) for i in range(1, max_layer_idx + 1)}
                sem_vis = create_color_visualization(semantic_map_warped, sem_colormap)
                path_sem_vis = sample_output_top_dir / "semantic_map_layer_idx_vis.png"
                save_image_data(sem_vis, path_sem_vis, 8)
                add_path('semantic_map_vis', path_sem_vis)
                
        # Save final combined actual mask if generated
        if final_combined_actual_mask is not None and out_opts.get('save_masks'):
            save_numpy(final_combined_actual_mask, sample_output_top_dir / "combined_actual_mask.npy") # etc.

        # Height Map
        if out_opts.get('save_topography_height_map', False) and topography_height_map is not None:
             path_hm = sample_output_top_dir / "topography_height_map.npy"
             save_numpy(topography_height_map, path_hm)
             add_path('height_map_npy', path_hm)
             if out_opts.get('save_visualizations'): hm_vis = normalize_image(topography_height_map)
             path_hm_vis = sample_output_top_dir / "topography_height_map_vis.png"
             save_image_data(hm_vis, path_hm_vis, 8)
             add_path('height_map_vis', path_hm_vis)


        # --- Artifact Maps ---
        if out_opts.get('save_warp_field') and warp_field_final is not None:
            path_wf = sample_output_top_dir / "warp_field.npy"
            save_numpy(warp_field_final, path_wf)
            add_path('warp_field_npy', path_wf)
            if out_opts.get('save_visualizations') and warp_field_vis_overlay is not None:
                path_wf_vis = sample_output_top_dir / "warp_field_vis.png"
                save_image_data(warp_field_vis_overlay, path_wf_vis, 8)
                add_path('warp_field_vis', path_wf_vis)
        if out_opts.get('save_noise_map') and total_added_noise is not None:
            path_nm = sample_output_top_dir / "noise_map_added.npy"
            save_numpy(total_added_noise, path_nm)
            add_path('noise_map_added_npy', path_nm)
            if out_opts.get('save_visualizations'):
                nm_vis = normalize_image(total_added_noise)
                path_nm_vis = sample_output_top_dir / "noise_map_added_vis.png"
                save_image_data(nm_vis, path_nm_vis, 8)
                add_path('noise_map_added_vis', path_nm_vis)

        # --- Overlays & GIFs ---
        if out_opts.get('save_overlays'):
            if overlay_contour_vis is not None: # Generated from final_combined_actual_mask (warped)
                path_ovc = sample_output_top_dir / "overlay_contour.png"; save_image_data(overlay_contour_vis, path_ovc, 8); add_path('overlay_contour', path_ovc)
            if instance_mask_vis_overlay is not None: # Generated from final_instance_mask (warped)
                path_ovi = sample_output_top_dir / "instance_mask_vis.png"; save_image_data(instance_mask_vis_overlay, path_ovi, 8); add_path('instance_mask_vis', path_ovi)
            if warp_field_vis_overlay is not None: # Generated from warp_field_final
                path_ovw = sample_output_top_dir / "warp_field_vis.png"; save_image_data(warp_field_vis_overlay, path_ovw, 8); add_path('warp_field_vis', path_ovw)
            if overlay_metadata_vis is not None:
                path_ovm = sample_output_top_dir / "overlay_metadata.png"; save_image_data(overlay_metadata_vis, path_ovm, 8); add_path('overlay_metadata', path_ovm)


        if out_opts.get('save_gifs') and actual_layer_gif_frames:
            path_gif = sample_output_top_dir / "layers_combined" / "layers_buildup.gif"
            save_gif_data(actual_layer_gif_frames, path_gif, duration=0.5)
            add_path('layers_actual_gif', path_gif)

        # --- Config & Metadata ---
        if out_opts.get('save_config'):
            path_cfg = sample_output_top_dir / "configuration_used.json"
            save_json_data(sample_config, path_cfg)
            add_path('configuration_used', path_cfg)
            metadata['output_paths'] = output_paths # Add file paths list
            path_meta = sample_output_top_dir / "metadata.json"
            save_json_data(metadata, path_meta)
            add_path('metadata', path_meta)

        # --- Hashes ---
        if out_opts.get('save_hashes'):
            hashes = calculate_hashes([p for p in save_paths_list if p.is_file()])
            path_hash = sample_output_top_dir / "hashes.json"
            save_json_data(hashes, path_hash) # Don't add hash path itself

        sample_logger.info(f"--- Successfully Completed Sample {sample_idx:05d} ---")
        return True

    # --- Main Exception Handler ---
    except Exception as e:
        sample_logger.error(f"!!! CRITICAL ERROR generating sample {sample_idx:05d} !!! Type: {type(e).__name__}, Error: {e}", exc_info=True)
        return False
    finally:
        # --- Close Logger Handler ---
        for handler in sample_logger.handlers[:]:
             if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_file_path):
                  try: handler.close()
                  except Exception: pass
             try: sample_logger.removeHandler(handler)
             except Exception: pass
        # ---

