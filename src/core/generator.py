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
                                                apply_local_affine, apply_shape_border,
                                                apply_corner_rounding)
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
    start_time = time.time()
    sample_rng = get_rng(sample_seed)

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
        sample_logger.setLevel(logging.getLogger().level) # Inherit level


    # --- Initialize ALL potential output variables to None outside the try block ---
    # This avoids UnboundLocalError if an error happens before assignment inside try
    image_final_noisy = None
    combined_mask_original_pre_warp = None
    semantic_map = None
    layer_order_map = None
    warp_field_final = None
    topography_height_map = None
    total_added_noise = None
    final_instance_mask = None
    final_instance_metadata = {}
    metadata = {}
    overlay_contour_vis, instance_mask_vis, warp_field_vis = None, None, None
    overlay_metadata_vis = None
    actual_layer_gif_frames = []
    output_paths = {}
    save_paths_list = []
    # ---
    
    try:
        # Config is already randomized by the calling function (task worker or CLI)
        sample_config = base_config
        logger.info(f"--- Generating Sample {sample_idx:05d} (Seed: {sample_seed}) ---")

        # --- Safely get image settings ---
        image_settings = sample_config.get('image_settings', {})
        h, w = image_settings.get('resolution', [256, 256]) # Default if missing
        bit_depth = image_settings.get('bit_depth', 16) # Default if missing
        magnification = image_settings.get('magnification', 1.0) # Default if missing
        pixel_size_nm_at_1x = image_settings.get('pixel_size_nm_at_1x', None) # Get safely
        pixel_size_nm = None
        if pixel_size_nm_at_1x is not None and magnification != 0:
            pixel_size_nm = pixel_size_nm_at_1x / magnification
        elif pixel_size_nm_at_1x is None:
            logger.warning(f"[Sample {sample_idx:05d}] 'pixel_size_nm_at_1x' not found in config.")
        elif magnification == 0: # Avoid division by zero
            logger.warning(f"[Sample {sample_idx:05d}] Magnification is zero, cannot calculate pixel size. Scale bar may be incorrect.")

        artifact_raffle_settings = sample_config.get('artifact_raffle', {})
        geom_mode = artifact_raffle_settings.get('per_instance_geometric_mode', 'none') # Default to none
        out_opts = sample_config.get('output_options', {})
        sample_logger.info(f"Image Size: {w}x{h}, Bit Depth: {bit_depth}, Magnification: {magnification:.2f}x")
        sample_logger.info(f"Per-instance Geometric Mode: {geom_mode}")

        # --- Initialize Raffler ---
        raff = Raffler(artifact_raffle_settings, sample_rng)
        
        # --- 2. Background ---
        bg_conf = sample_config.get('background', {})
        background_clean = generate_background(bg_conf, (h, w), magnification, sample_rng)
        initial_background = background_clean.copy()
        image_clean = background_clean.copy() # Composition starts here
        
        # --- 3. Layers ---
        layering_conf = sample_config.get('layering', {})
        selected_layers = layering_conf.get('selected_layers', []) # Should be list of dicts
        composition_mode = layering_conf.get('composition_mode', 'additive')
        randomize_layer_order = layering_conf.get('randomize_order', False)
        layer_indices = list(range(len(selected_layers)))
        if randomize_layer_order: sample_rng.shuffle(layer_indices)
        layer_colors = get_distinct_colors(len(selected_layers))
        layer_id_to_color = {i: layer_colors[i % len(layer_colors)] for i in range(len(selected_layers))}
        
        all_layers_data = {}
        layer_masks_original = {} # Store original combined masks per layer
        layer_masks_actual = {}   # Store final combined masks AFTER layer-wide etch bias
        layer_renders_actual = [] # Final float renders per layer (for composition)
        layer_renders_clean = []  # Clean renders per layer (before render artifacts)
        instance_artifact_params_list = [] # Details of artifacts applied per instance/layer
        # Initialize maps needed for outputs (warp applies to these)
        layer_defect_masks = {}
        layer_order_map_pre_warp = np.zeros((h, w), dtype=np.uint8) if out_opts.get('save_layer_order_map') else None
        semantic_map_pre_warp = np.zeros((h, w), dtype=np.uint8) if out_opts.get('save_semantic_map') else None
        
        # --- 3a. Layer Generation Loop ---
        sample_logger.info(f"Processing {len(selected_layers)} selected layers...")
        for layer_render_idx, layer_config_idx in enumerate(layer_indices):
            layer_conf = selected_layers[layer_config_idx]
            layer_id_str = f"layer_{layer_config_idx:02d}"
            layer_id_name = layer_conf.get('layer_id', layer_id_str)
            layer_output_dir = sample_output_top_dir / "layers" / layer_id_str
            ensure_dir(layer_output_dir)
            sample_logger.info(f" -> Layer {layer_render_idx+1}/{len(layer_indices)} (Config Idx: {layer_config_idx}): '{layer_id_name}'")

            # --- Safely get parameters for this layer ---
            intensity = layer_conf.get('intensity', 0.5)
            shape_type = layer_conf.get('shape', 'circle') # Already resolved from _choices
            alpha = layer_conf.get('alpha', 1.0)
            shape_params_base = layer_conf.get('shape_params', {})
            pattern_params = layer_conf.get('pattern_params', {})
            pattern_type = layer_conf.get('pattern', 'grid')

            # Raffle artifact definitions ONCE for this layer
            layer_shape_artifacts_defs = raff.raffle_effects('shape')
            sample_logger.debug(f"    Raffled shape artifacts for layer: {[a.get('name', 'N/A') for a in layer_shape_artifacts_defs]}")

            # Pre-generate layer-wide noise map if local_brightness might be applied
            brightness_artifact_def = next((a for a in layer_shape_artifacts_defs if a.get('name') == 'local_brightness'), None)
            layer_brightness_noise_map = None

            layer_brightness_noise_map = None
            if brightness_artifact_def and HAS_PERLIN and out_opts.get('include_artifact_params_in_metadata', False): # Only gen if needed
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

            # Get positions or paths for this layer
            positions_or_paths = get_pattern_positions(pattern_type, (h, w), shape_params_base, pattern_params, sample_rng)
            is_path_based = pattern_type in ['sine_wave_horizontal']
            sample_logger.debug(f"    Generated {len(positions_or_paths)} positions/paths for pattern '{pattern_type}'")

            # Initialize accumulators for this layer
            layer_combined_mask_original_acc = np.zeros((h, w), dtype=np.uint8)
            layer_combined_mask_actual_acc = np.zeros((h, w), dtype=np.uint8) # Before layer etch bias
            layer_render_buffer_final_acc = np.zeros((h, w), dtype=np.float32) # With render artifacts
            layer_render_buffer_clean_acc = np.zeros((h, w), dtype=np.float32) # Before render artifacts
            num_instances_in_layer = 0
            applied_shape_artifacts_names_layer = set()

            # --- 3b. Instance/Path Loop ---
            for idx, item in enumerate(positions_or_paths):
                shape_params = copy.deepcopy(shape_params_base) # Use deep copy for safety
                
                # --- Declare variables used in both branches ---
                original_mask_instance = None
                actual_mask_instance = None
                instance_render_clean = None
                instance_render_final = None

                # --- A. Path Processing ---
                if is_path_based:
                    path_points = item
                    if not path_points or len(path_points) < 2: continue
                    thickness = max(1, int(round(shape_params.get('thickness', 2))))

                    # 1. Create initial mask by drawing path
                    temp_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.polylines(temp_mask, [np.array(path_points, dtype=np.int32)], isClosed=False, color=1, thickness=thickness)
                    original_mask_instance = temp_mask

                # --- B. Position Processing ---
                else:
                    pos = item
                    # Set shape_params center or endpoints
                    if shape_type.endswith('line') and isinstance(pos, tuple) and len(pos) == 2 and isinstance(pos[0], tuple):
                         shape_params['x1'], shape_params['y1'] = pos[0];
                         shape_params['x2'], shape_params['y2'] = pos[1]
                    elif isinstance(pos, tuple) and len(pos) == 2 and isinstance(pos[0], (int, float)):
                         shape_params['center_x'], shape_params['center_y'] = pos
                    else:
                        sample_logger.warning(f"Unsupported position format for shape {shape_type}: {pos}. Skipping instance.")
                        continue
                    # 1. Create mask
                    original_mask_instance = create_shape_mask(shape_type, shape_params, (h, w), rng=sample_rng)

                # --- Common Instance Processing Steps ---
                if original_mask_instance is None or np.sum(original_mask_instance) == 0:
                    continue
                layer_combined_mask_original_acc |= original_mask_instance # Accumulate original

                # 2. Apply MASK Artifacts (with per-instance variation)
                actual_mask_instance = original_mask_instance.copy()
                applied_mask_artifacts_instance_names = []
                for layer_artifact_def in layer_shape_artifacts_defs:
                    artifact_name = layer_artifact_def.get('name')
                    if not artifact_name or artifact_name in SHAPE_RENDER_ARTIFACT_FUNCS or artifact_name == 'etch_bias': continue

                    # Vary params
                    layer_params = layer_artifact_def.get('params', {})
                    instance_params = {}
                    variation_factor = 0.2
                    affine_variation_factor = 0.1 # Example factors
                    current_variation = variation_factor if artifact_name != 'local_affine' else affine_variation_factor
                    for p_name, p_val in layer_params.items():
                        if isinstance(p_val, (int, float)):
                            offset = p_val * current_variation * sample_rng.uniform(-1.0, 1.0)
                            if isinstance(p_val, int):
                                instance_params[p_name] = max(0, int(round(p_val + offset)))
                            else:
                                instance_params[p_name] = p_val + offset
                        else:
                            instance_params[p_name] = p_val

                    # Check mode & get func
                    apply_this_artifact = True
                    target_mask_artifact_func = None
                    if artifact_name == 'local_elastic':
                        if geom_mode != 'local_elastic':
                            apply_this_artifact = False
                        else: target_mask_artifact_func = SHAPE_MASK_ARTIFACT_FUNCS.get(artifact_name)
                    elif artifact_name == 'local_affine':
                        if geom_mode != 'local_affine':
                            apply_this_artifact = False
                        else:
                            target_mask_artifact_func = SHAPE_MASK_ARTIFACT_FUNCS.get(artifact_name)
                    elif artifact_name in SHAPE_MASK_ARTIFACT_FUNCS:
                        target_mask_artifact_func = SHAPE_MASK_ARTIFACT_FUNCS.get(artifact_name)
                    else:
                        apply_this_artifact = False

                    # Apply
                    if apply_this_artifact and target_mask_artifact_func:
                        try:
                            actual_mask_instance = target_mask_artifact_func(actual_mask_instance, instance_params, sample_rng)
                            applied_mask_artifacts_instance_names.append(artifact_name)
                            applied_shape_artifacts_names_layer.add(artifact_name)
                            if out_opts.get('include_artifact_params_in_metadata', False):
                                instance_artifact_params_list.append({ 
                                    'sample_idx': sample_idx,
                                    'instance_idx_in_layer': idx,
                                    'layer_config_idx': layer_config_idx,
                                    'artifact_name': artifact_name,
                                    'parameters': copy.deepcopy(instance_params) })
                        except Exception as e:
                            sample_logger.error(f"Error applying mask artifact {artifact_name} to instance {idx}: {e}", exc_info=True)

                # 3. Initial CLEAN Render (based on FINAL mask)
                instance_render_clean = np.zeros_like(layer_render_buffer_final_acc)
                if is_path_based: # Draw path directly if path based
                    pts_np = np.array([item], dtype=np.int32) # item is path_points
                    cv2.polylines(instance_render_clean, pts_np, isClosed=False, color=(intensity*alpha), thickness=max(1, int(round(shape_params.get('thickness', 2)))))
                    # Apply mask just in case polyline went slightly out
                    instance_render_clean[actual_mask_instance == 0] = 0
                else: # Render shape
                    temp_render_geometry = render_shape(shape_type, shape_params, (h, w), intensity=1.0, anti_aliasing=True, rng=sample_rng)
                    instance_render_clean[actual_mask_instance > 0] = temp_render_geometry[actual_mask_instance > 0] * intensity * alpha
                instance_render_clean = np.clip(instance_render_clean, 0.0, 1.0)
                layer_render_buffer_clean_acc += instance_render_clean # Accumulate clean render

                # 4. Apply RENDER Artifacts to get FINAL render
                instance_render_final = instance_render_clean.copy() # Start from clean
                applied_render_artifacts_instance_names = []
                render_artifact_order = ['shape_border', 'local_brightness']
                for artifact_name in render_artifact_order:
                    render_artifact_def = next((a for a in layer_shape_artifacts_defs if a.get('name') == artifact_name), None)
                    if render_artifact_def:
                        target_render_artifact_func = SHAPE_RENDER_ARTIFACT_FUNCS.get(artifact_name)
                        if target_render_artifact_func:
                            try:
                                # Vary params for render artifact
                                layer_params_render = render_artifact_def.get('params', {})
                                instance_params_render = {} # ... (code to vary params) ...
                                variation_factor = 0.2
                                for p_name, p_val in layer_params_render.items():
                                    if isinstance(p_val, (int, float)):
                                        offset = p_val * variation_factor * sample_rng.uniform(-1.0, 1.0)
                                        if p_name == 'thickness': # Ensure border thickness >= 1
                                             instance_params_render[p_name] = max(1, int(round(p_val + offset)))
                                        elif isinstance(p_val, int): instance_params_render[p_name] = max(0, int(round(p_val + offset)))
                                        else: instance_params_render[p_name] = p_val + offset
                                    else: instance_params_render[p_name] = p_val

                                # Apply render artifact
                                if artifact_name == 'shape_border':
                                    instance_render_final = target_render_artifact_func(instance_render_final, actual_mask_instance, instance_params_render, intensity, sample_rng)
                                elif artifact_name == 'local_brightness':
                                    if layer_brightness_noise_map is not None:
                                        instance_contrast = instance_params_render.get('contrast', 0.1)
                                        noise_slice = layer_brightness_noise_map[actual_mask_instance > 0]
                                        if noise_slice.size > 0: # Ensure mask is not empty
                                            brightness_variation = noise_slice * instance_contrast
                                            instance_render_final[actual_mask_instance > 0] *= (1.0 + brightness_variation)
                                            instance_render_final = np.clip(instance_render_final, 0.0, 1.0)
                                    else: sample_logger.warning("Skipping local_brightness (no noise map).")

                                applied_render_artifacts_instance_names.append(artifact_name)
                                applied_shape_artifacts_names_layer.add(artifact_name)
                                if out_opts.get('include_artifact_params_in_metadata', False):
                                     instance_artifact_params_list.append({ 'sample_idx': sample_idx, 'instance_idx_in_layer': idx, 'layer_config_idx': layer_config_idx, 'artifact_name': artifact_name, 'parameters': copy.deepcopy(instance_params_render) })

                            except Exception as e: sample_logger.error(f"Error applying render artifact {artifact_name} to instance {idx}: {e}", exc_info=True)

                # 5. Accumulate Masks and Renders
                layer_combined_mask_actual_acc |= actual_mask_instance # Accumulate mask BEFORE layer etch bias
                layer_render_buffer_final_acc += instance_render_final # Accumulate final render
                num_instances_in_layer += 1
            # --- End Instance/Path Loop ---

            # --- Apply Layer-Wide Etch Bias ---
            layer_combined_mask_final_actual = layer_combined_mask_actual_acc # Start with accumulated mask
            etch_bias_def = next((a for a in layer_shape_artifacts_defs if a.get('name') == 'etch_bias'), None)
            if etch_bias_def:
                 try:
                    layer_etch_params = etch_bias_def.get('params', {})
                    layer_combined_mask_final_actual = apply_etch_bias(layer_combined_mask_final_actual, layer_etch_params, sample_rng)
                    applied_shape_artifacts_names_layer.add('etch_bias')
                    if out_opts.get('include_artifact_params_in_metadata', False):
                        instance_artifact_params_list.append({
                            'sample_idx': sample_idx,
                            'instance_idx_in_layer': -1,
                            'layer_config_idx': layer_config_idx,
                            'artifact_name': 'etch_bias',
                            'parameters': copy.deepcopy(layer_etch_params)})
                 except Exception as e_etch:
                     sample_logger.error(f"Error applying etch_bias to layer {layer_config_idx}: {e_etch}", exc_info=True)
            layer_masks_actual[layer_config_idx] = layer_combined_mask_final_actual # Store FINAL actual mask

            # --- Update Global Maps (using FINAL actual mask) ---
            if layer_order_map is not None:
                layer_order_map[layer_combined_mask_final_actual > 0] = layer_render_idx + 1
            if semantic_map_pre_warp is not None:
                semantic_map_pre_warp[layer_combined_mask_final_actual > 0] = layer_config_idx + 1

            # --- Final Layer Clipping & Storage ---
            layer_render_buffer_final_acc = np.clip(layer_render_buffer_final_acc, 0.0, 1.0)
            layer_render_buffer_clean_acc = np.clip(layer_render_buffer_clean_acc, 0.0, 1.0)
            layer_renders_actual.append(layer_render_buffer_final_acc)
            layer_renders_clean.append(layer_render_buffer_clean_acc)

            # --- Store Layer Data ---
            defect_mask_layer = generate_defect_mask(layer_combined_mask_original_acc, layer_combined_mask_final_actual)
            all_layers_data[layer_config_idx] = {
                 'id': layer_id_name,
                 'original_mask': layer_combined_mask_original_acc.copy(), # Store copy
                 'actual_mask': layer_combined_mask_final_actual.copy(),   # Store copy
                 'defect_mask': defect_mask_layer,
                 'render_clean': layer_render_buffer_clean_acc.copy() if out_opts.get('save_clean_layer_renders') else None,
                 'render_final': layer_render_buffer_final_acc.copy() if out_opts.get('save_per_layer_renders') else None,
                 'output_dir': layer_output_dir,
                 'num_instances': num_instances_in_layer,
                 'applied_shape_artifacts': sorted(list(applied_shape_artifacts_names_layer))
            }
            layer_masks_original[layer_config_idx] = layer_combined_mask_original_acc # Keep for combined original mask
            if defect_mask_layer is not None:
                layer_defect_masks[layer_config_idx] = defect_mask_layer

            # --- Save Per-Layer Outputs ---
            save_paths_list_layer = []
            def add_layer_path(key, path_obj):
                save_paths_list_layer.append(path_obj) # Helper if needed later
            # Save masks
            if out_opts.get('save_masks'):
                 save_numpy(all_layers_data[layer_config_idx]['original_mask'], layer_output_dir / "original_mask.npy")
                 save_numpy(all_layers_data[layer_config_idx]['actual_mask'], layer_output_dir / "actual_mask.npy")
                 if out_opts.get('save_defect_masks') and defect_mask_layer is not None:
                      save_numpy(defect_mask_layer, layer_output_dir / "defect_mask.npy")
            # Save visualizations
            if out_opts.get('save_visualizations'):
                 save_image_data(all_layers_data[layer_config_idx]['original_mask'].astype(float), layer_output_dir / "original_mask_vis.png", 8)
                 save_image_data(all_layers_data[layer_config_idx]['actual_mask'].astype(float), layer_output_dir / "actual_mask_vis.png", 8)
                 if out_opts.get('save_defect_masks') and defect_mask_layer is not None:
                      save_image_data(defect_mask_layer.astype(float), layer_output_dir / "defect_mask_vis.png", 8)
            # Save renders
            if out_opts.get('save_clean_layer_renders') and all_layers_data[layer_config_idx]['render_clean'] is not None:
                 path_render_clean = layer_output_dir / "render_clean_vis.png"
                 save_image_data(all_layers_data[layer_config_idx]['render_clean'], path_render_clean, bit_depth)
            if out_opts.get('save_per_layer_renders') and all_layers_data[layer_config_idx]['render_final'] is not None:
                 path_render_final = layer_output_dir / "render_final_vis.png"
                 save_image_data(all_layers_data[layer_config_idx]['render_final'], path_render_final, bit_depth)
        # --- End Layer Loop ---
        sample_logger.info("Finished processing all layers.")
        
        
        # --- 4. Compose Layers ---
        sample_logger.info(f"Composing {len(layer_renders_actual)} layers using mode: {composition_mode}")
        cumulative_layers_for_gif = [initial_background.copy()]
        for layer_buffer in layer_renders_actual:
             if composition_mode == 'additive': image_clean += layer_buffer
             elif composition_mode == 'multiplicative': image_clean *= (1.0 + layer_buffer * 2) # Example multiplicative blend
             elif composition_mode == 'overwrite':
                   current_layer_config_idx = layer_indices[len(cumulative_layers_for_gif)-1] # Index of layer config for this buffer
                   mask = layer_masks_actual.get(current_layer_config_idx) # Use FINAL actual mask
                   if mask is not None: image_clean[mask > 0] = layer_buffer[mask > 0]
                   else: sample_logger.warning(f"Mask not found for layer {current_layer_config_idx} during overwrite composition.")
             else: image_clean += layer_buffer # Default additive
             image_clean = np.clip(image_clean, 0.0, 1.0)
             cumulative_layers_for_gif.append(image_clean.copy())


        image_clean_pre_warp = image_clean.copy()
        sample_logger.debug("Layer composition complete.")

        # --- 5. Generate Combined Original Mask (Pre-Warp) ---
        combined_mask_original_pre_warp = generate_combined_mask(layer_masks_original)

        # --- 6. Global Geometric Warping ---
        sample_logger.info("Applying global geometric artifacts...")
        margin_factor = sample_config['image_settings']['oversize_margin_factor']
        margin_h, margin_w = int(h * margin_factor), int(w * margin_factor)
        oversized_h, oversized_w = h + 2 * margin_h, w + 2 * margin_w
        oversized_shape = (oversized_h, oversized_w)

        # Helper to embed (can be defined locally or imported from utils if preferred)
        def embed_in_oversized(img, target_shape, margin_h, margin_w, border_mode=cv2.BORDER_REFLECT_101, border_value=0):
            if img is None: return None
            try: # Add try/except for safety
                # Use copyMakeBorder for cleaner padding
                padded = cv2.copyMakeBorder(img, margin_h, margin_h, margin_w, margin_w, border_mode, value=border_value)
                # Ensure exact shape due to potential rounding issues
                return padded[:target_shape[0], :target_shape[1]]
            except Exception as e_embed:
                 sample_logger.error(f"Error embedding image/mask into oversized canvas: {e_embed}", exc_info=True)
                 # Return an empty canvas of the target shape/dtype as fallback
                 return np.full(target_shape, border_value, dtype=img.dtype)


        # Embed data needed for warping
        image_clean_oversized = embed_in_oversized(image_clean_pre_warp, oversized_shape, margin_h, margin_w)
        semantic_map_oversized = embed_in_oversized(semantic_map_pre_warp, oversized_shape, margin_h, margin_w, cv2.BORDER_CONSTANT, 0) if semantic_map_pre_warp is not None else None
        layer_order_map_oversized = embed_in_oversized(layer_order_map_pre_warp, oversized_shape, margin_h, margin_w, cv2.BORDER_CONSTANT, 0) if layer_order_map_pre_warp is not None else None

        # Apply geometric artifacts (loop through raffled effects)
        geometric_artifacts = raff.raffle_effects('geometric')
        warp_field_combined = None
        maps_to_warp = [m for m in [semantic_map_oversized, layer_order_map_oversized] if m is not None]
        applied_geometric_artifacts = []

        for artifact in geometric_artifacts:
             func = GEOMETRIC_ARTIFACT_FUNCS.get(artifact['name'])
             if func:
                  try:
                       sample_logger.info(f"Applying geometric artifact: {artifact['name']}")
                       image_clean_oversized, warped_maps_out, warp_field = \
                           func(image_clean_oversized, maps_to_warp, artifact['params'], sample_rng)
                       if warped_maps_out: maps_to_warp = warped_maps_out # Update for next warp step
                       if warp_field is not None: warp_field_combined = warp_field # Store last warp field
                       applied_geometric_artifacts.append(artifact['name'])
                  except Exception as e_geo_art: sample_logger.error(f"Error applying geometric artifact {artifact['name']}: {e_geo_art}", exc_info=True)
             else: sample_logger.warning(f"Geometric artifact function '{artifact['name']}' not found.")

        # Update maps after warping
        semantic_map_warped = None; layer_order_map_warped = None
        map_idx = 0
        if semantic_map_oversized is not None: semantic_map_warped = maps_to_warp[map_idx]; map_idx += 1
        if layer_order_map_oversized is not None: layer_order_map_warped = maps_to_warp[map_idx]; map_idx += 1

        # Center Crop Back
        image_clean_warped = image_clean_oversized[margin_h:margin_h+h, margin_w:margin_w+w]
        semantic_map = semantic_map_warped[margin_h:margin_h+h, margin_w:margin_w+w] if semantic_map_warped is not None else None
        layer_order_map = layer_order_map_warped[margin_h:margin_h+h, margin_w:margin_w+w] if layer_order_map_warped is not None else None
        warp_field_final = warp_field_combined[margin_h:margin_h+h, margin_w:margin_w+w] if warp_field_combined is not None else None
        image_post_geometric_warp = image_clean_warped.copy() # Capture state here
        sample_logger.debug("Geometric warping complete.")

        # --- 7. Save Post-Geometric Intermediate ---
        if out_opts.get('save_extra_intermediate', False):
            path_post_geo = sample_output_top_dir / "image_post_geometric_warp.png"
            save_image_data(image_post_geometric_warp, path_post_geo, 8)

        # --- 8. Apply Instrument Artifacts ---
        sample_logger.info("Applying instrument artifacts...")
        image_post_instrument = image_post_geometric_warp
        applied_instrument_artifacts = []
        total_added_fpn = np.zeros_like(image_post_instrument)

        for artifact in raff.raffle_effects('instrument'): # Raffle instrument effects
            func = INSTRUMENT_ARTIFACT_FUNCS.get(artifact['name'])
            if func:
                try:
                    sample_logger.info(f"Applying instrument artifact: {artifact['name']}")
                    params = artifact.get('params', {})
                    if artifact['name'] == 'topographic_shading':
                         image_post_instrument, height_map_generated = func(image_post_instrument, all_layers_data, params, sample_rng)
                         if out_opts.get('save_topography_height_map', False): topography_height_map = height_map_generated
                    elif artifact['name'] == 'fixed_pattern_noise':
                         image_post_instrument, fpn_map = func(image_post_instrument, params, sample_rng)
                         total_added_fpn += fpn_map
                    else: # Other instrument effects (psf, defocus, charging, edge_brightness, etc.)
                         image_post_instrument = func(image_post_instrument, params, sample_rng)
                    applied_instrument_artifacts.append(artifact['name'])
                except Exception as e_inst_art: sample_logger.error(f"Error applying instrument artifact {artifact['name']}: {e_inst_art}", exc_info=True)
            else: sample_logger.warning(f"Instrument artifact function '{artifact['name']}' not found.")

        image_post_instrument = np.clip(image_post_instrument, 0.0, 1.0)
        sample_logger.debug("Instrument artifacts complete.")

         # --- 9. Save Post-Instrument Intermediate ---
        if out_opts.get('save_extra_intermediate', False):
             path_post_inst = sample_output_top_dir / "image_post_instrument.png"
             save_image_data(image_post_instrument, path_post_inst, 8)

        # --- 10. Apply Detector Noise ---
        sample_logger.info("Applying detector noise...")
        image_final_noisy = image_post_instrument.copy()
        total_added_noise = total_added_fpn.copy()
        applied_noise_artifacts = []
        noise_artifacts = raff.raffle_effects('noise')
        # Apply quantization first if present
        quant_artifact = next((a for a in noise_artifacts if a.get('name') == 'quantization'), None)
        if quant_artifact:
             try:
                  logger.info(f"Applying noise: {quant_artifact['name']}")
                  image_final_noisy, q_noise = apply_noise(image_final_noisy, quant_artifact['name'], quant_artifact['params'], sample_rng)
                  total_added_noise += q_noise
                  applied_noise_artifacts.append(quant_artifact['name'])
             except Exception as e: sample_logger.error(f"Error applying quantization: {e}", exc_info=True)
        # Apply other noise types
        for artifact in noise_artifacts:
             if artifact.get('name') == 'quantization': continue
             noise_type = artifact.get('name')
             if not noise_type: continue
             try:
                 sample_logger.info(f"Applying noise: {noise_type}")
                 image_final_noisy, added_noise = apply_noise(image_final_noisy, noise_type, artifact['params'], sample_rng)
                 if added_noise is not None: total_added_noise += added_noise
                 applied_noise_artifacts.append(noise_type)
             except Exception as e_noise: sample_logger.error(f"Error applying noise {noise_type}: {e_noise}", exc_info=True)
        image_final_noisy = np.clip(image_final_noisy, 0.0, 1.0)
        sample_logger.debug("Detector noise complete.")

        # --- 11. Generate Final Instance GT from Warped Data ---
        sample_logger.info("Generating final instance data...")
        if out_opts.get('save_bounding_boxes', False):
            if semantic_map is not None:
                 try:
                    # Use generate_instance_data with warped map
                    final_instance_mask, final_instance_metadata = generate_instance_data(
                         layer_masks_actual={}, # Pass empty dict as layer masks aren't warped individually
                         warped_semantic_map=semantic_map # Pass warped map
                    )
                    sample_logger.info(f"Generated {len(final_instance_metadata)} instance annotations from warped semantic map.")
                 except Exception as e_inst:
                      sample_logger.error(f"Error generating instance data from warped map: {e_inst}", exc_info=True)
                      final_instance_metadata = {'error': "Failed to generate instance data post-warp."} # Store error
            else:
                 final_instance_metadata = {'placeholder': "Cannot generate instance data without semantic map."}


        # --- 12. Generate Overlays ---
        sample_logger.info("Generating overlays...")
        final_image_vis_8bit = image_to_bit_depth(image_final_noisy, 8)
        try:
            # Use final_instance_mask if generated, else None
             overlay_contour_vis, instance_mask_vis_maybe, warp_field_vis = create_overlays(
                 final_image_vis_8bit,
                 (final_instance_mask > 0).astype(np.uint8) if final_instance_mask is not None else None, # Create combined mask from instance mask
                 final_instance_mask,
                 warp_field_final,
                 layer_id_to_color
             )
             # Check if instance_mask_vis was actually generated
             instance_mask_vis = instance_mask_vis_maybe
        except Exception as e_overlay:
             sample_logger.error(f"Error generating overlays: {e_overlay}", exc_info=True)

        metadata_text = f"Sample: {sample_idx:05d}\nSeed: {sample_seed}\nMag: {magnification:.2f}x"
        overlay_metadata_vis = add_metadata_overlay(final_image_vis_8bit, text=metadata_text, pixel_size_nm=pixel_size_nm)

        if out_opts.get('save_gifs') and cumulative_layers_for_gif:
             for frame in cumulative_layers_for_gif:
                  actual_layer_gif_frames.append(image_to_bit_depth(frame, 8))
    
    
        # --- 13. Prepare Metadata ---
        metadata = {
            "sample_index": sample_idx,
            "sample_name": f"sem_{sample_idx:05d}",
            "seed": sample_seed,
            "resolution": [h, w],
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
        def add_path(key, path_obj):
            if path_obj and path_obj.exists(): # Check if file exists before adding
                try:
                    output_paths[key] = str(path_obj.relative_to(output_parent_dir))
                    save_paths_list.append(path_obj)
                except ValueError: # Handle case where path might not be relative (e.g. different drive)
                     output_paths[key] = str(path_obj)
                     save_paths_list.append(path_obj)
            elif path_obj:
                 sample_logger.warning(f"File path added to metadata does not exist: {path_obj}")
                 output_paths[key] = f"MISSING: {path_obj.name}"

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
             save_image_data(image_clean_pre_warp, path, 8); add_path('image_clean_pre_warp', path)
        if out_opts.get('save_extra_intermediate', False):
             path_post_geo = sample_output_top_dir / "image_post_geometric_warp.png"
             save_image_data(image_post_geometric_warp, path_post_geo, 8); add_path('image_post_geometric_warp', path_post_geo)
             path_post_inst = sample_output_top_dir / "image_post_instrument.png"
             save_image_data(image_post_instrument, path_post_inst, 8); add_path('image_post_instrument', path_post_inst)
        path_final_vis = sample_output_top_dir / "image_final_noisy_vis.png"
        save_image_data(image_final_noisy, path_final_vis, 8); add_path('image_final_noisy_vis', path_final_vis)
        # Background
        path_bg_npy = sample_output_top_dir / "layers_combined" / "background.npy"; save_numpy(initial_background, path_bg_npy); add_path('background_npy', path_bg_npy)
        path_bg_vis = sample_output_top_dir / "layers_combined" / "background_vis.png"; save_image_data(initial_background, path_bg_vis, 8); add_path('background_vis', path_bg_vis)


        # --- Masks and GT Maps ---
        if out_opts.get('save_masks'):
            # Combined Original (Pre-Warp)
            if combined_mask_original_pre_warp is not None:
                path = sample_output_top_dir / "combined_original_mask.npy"
                save_numpy(combined_mask_original_pre_warp, path)
                add_path('combined_original_mask_npy', path)
                if out_opts.get('save_visualizations'):
                    path_vis = sample_output_top_dir / "combined_original_mask_vis.png"
                    save_image_data(combined_mask_original_pre_warp.astype(float), path_vis, 8)
                    add_path('combined_original_mask_vis', path_vis)
            # Final Combined Actual Mask (from final instance mask)
            if final_instance_mask is not None:
                final_combined_actual = (final_instance_mask > 0).astype(np.uint8)
                path = sample_output_top_dir / "combined_actual_mask.npy"
                save_numpy(final_combined_actual, path)
                add_path('combined_actual_mask_npy', path)
                if out_opts.get('save_visualizations'):
                    path_vis = sample_output_top_dir / "combined_actual_mask_vis.png"
                    save_image_data(final_combined_actual.astype(float), path_vis, 8)
                    add_path('combined_actual_mask_vis', path_vis)

             # Final Instance Mask
            if final_instance_mask is not None:
                inst_mask_format = out_opts.get('output_formats', {}).get('instance_mask', 'tif').lower()
                path_inst = sample_output_top_dir / f"instance_mask.{inst_mask_format}"
                save_image_data(final_instance_mask, path_inst, bit_depth, format_hint=inst_mask_format.upper())
                add_path('instance_mask', path_inst) # Add final path used
                # Instance vis saved below in overlays

        # Layer Order Map
        if out_opts.get('save_layer_order_map', False) and layer_order_map is not None:
             path_lom = sample_output_top_dir / "layer_order_map.npy"; save_numpy(layer_order_map.astype(np.uint8), path_lom); add_path('layer_order_map_npy', path_lom)
             if out_opts.get('save_visualizations'):
                  lom_vis_gray = normalize_image(layer_order_map); path_lom_vis = sample_output_top_dir / "layer_order_map_vis.png"; save_image_data(lom_vis_gray, path_lom_vis, 8); add_path('layer_order_map_vis', path_lom_vis)
        # Semantic Map
        if out_opts.get('save_semantic_map', False) and semantic_map is not None:
             path_sem = sample_output_top_dir / "semantic_map_layer_idx.npy"; save_numpy(semantic_map.astype(np.uint8), path_sem); add_path('semantic_map_npy', path_sem)
             if out_opts.get('save_visualizations'):
                 max_layer_idx = np.max(semantic_map); sem_colormap = {i: layer_id_to_color.get(i-1, (128,128,128)) for i in range(1, max_layer_idx + 1)}; sem_vis = create_color_visualization(semantic_map, sem_colormap); path_sem_vis = sample_output_top_dir / "semantic_map_layer_idx_vis.png"; save_image_data(sem_vis, path_sem_vis, 8); add_path('semantic_map_vis', path_sem_vis)
        # Height Map
        if out_opts.get('save_topography_height_map', False) and topography_height_map is not None:
             path_hm = sample_output_top_dir / "topography_height_map.npy"; save_numpy(topography_height_map, path_hm); add_path('height_map_npy', path_hm)
             if out_opts.get('save_visualizations'): hm_vis = normalize_image(topography_height_map); path_hm_vis = sample_output_top_dir / "topography_height_map_vis.png"; save_image_data(hm_vis, path_hm_vis, 8); add_path('height_map_vis', path_hm_vis)


        # --- Artifact Maps ---
        if out_opts.get('save_warp_field') and warp_field_final is not None:
             path_wf = sample_output_top_dir / "warp_field.npy"; save_numpy(warp_field_final, path_wf); add_path('warp_field_npy', path_wf)
             if out_opts.get('save_visualizations') and warp_field_vis is not None: path_wf_vis = sample_output_top_dir / "warp_field_vis.png"; save_image_data(warp_field_vis, path_wf_vis, 8); add_path('warp_field_vis', path_wf_vis)
        if out_opts.get('save_noise_map') and total_added_noise is not None:
             path_nm = sample_output_top_dir / "noise_map_added.npy"; save_numpy(total_added_noise, path_nm); add_path('noise_map_added_npy', path_nm)
             if out_opts.get('save_visualizations'): nm_vis = normalize_image(total_added_noise); path_nm_vis = sample_output_top_dir / "noise_map_added_vis.png"; save_image_data(nm_vis, path_nm_vis, 8); add_path('noise_map_added_vis', path_nm_vis)

        # --- Overlays & GIFs ---
        if out_opts.get('save_overlays'):
             if overlay_contour_vis is not None: path_ovc = sample_output_top_dir / "overlay_contour.png"; save_image_data(overlay_contour_vis, path_ovc, 8); add_path('overlay_contour', path_ovc)
             if instance_mask_vis is not None: path_ovi = sample_output_top_dir / "instance_mask_vis.png"; save_image_data(instance_mask_vis, path_ovi, 8); add_path('instance_mask_vis', path_ovi) # Save instance vis here
             if overlay_metadata_vis is not None: path_ovm = sample_output_top_dir / "overlay_metadata.png"; save_image_data(overlay_metadata_vis, path_ovm, 8); add_path('overlay_metadata', path_ovm)
        if out_opts.get('save_gifs') and actual_layer_gif_frames:
             path_gif = sample_output_top_dir / "layers_combined" / "layers_buildup.gif"; save_gif_data(actual_layer_gif_frames, path_gif, duration=0.5); add_path('layers_actual_gif', path_gif)

        # --- Config & Metadata ---
        if out_opts.get('save_config'): path_cfg = sample_output_top_dir / "configuration_used.json"; save_json_data(sample_config, path_cfg); add_path('configuration_used', path_cfg)
        metadata['output_paths'] = output_paths # Add file paths list
        path_meta = sample_output_top_dir / "metadata.json"; save_json_data(metadata, path_meta); add_path('metadata', path_meta)

        # --- Hashes ---
        if out_opts.get('save_hashes'): hashes = calculate_hashes([p for p in save_paths_list if p.is_file()]); path_hash = sample_output_top_dir / "hashes.json"; save_json_data(hashes, path_hash) # Don't add hash path itself

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

