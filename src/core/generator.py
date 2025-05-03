import numpy as np
import cv2
import os
import time
import logging
import imageio
from pathlib import Path

from .utils import get_rng, ensure_dir, normalize_image, image_to_bit_depth, get_distinct_colors
from .configuration import randomize_config_for_sample
from ..components.raffler import Raffler
from ..components.background import generate_background
from ..components.patterns import get_pattern_positions
from ..components.shapes import create_shape_mask, render_shape, draw_shape
from ..components.noise import apply_noise
from ..outputs.writers import (save_numpy, save_image_data, save_json_data,
                             save_gif_data, save_text_file, calculate_hashes)
from ..outputs.ground_truth import (generate_instance_mask, generate_combined_mask,
                                  create_overlays, add_metadata_overlay)


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
# Import specific artifact functions from their new modules
from ..components.artifacts.shape_level import (apply_edge_ripple, apply_breaks_holes,
                                                apply_local_elastic, apply_contour_smoothing,
                                                apply_local_brightness)
from ..components.artifacts.geometric import apply_affine, apply_elastic
from ..components.artifacts.instrument_optical import (apply_psf_blur, apply_defocus_blur,
                                                     apply_charging, apply_topographic_shading,
                                                     apply_gradient_illumination, apply_striping_smearing,
                                                     apply_fixed_pattern_noise)
from ..components.noise import apply_noise
from ..outputs.writers import (save_numpy, save_image_data, save_json_data,
                             save_gif_data, save_text_file, calculate_hashes)
from ..outputs.ground_truth import (generate_instance_mask, generate_combined_mask,
                                  generate_defect_mask, create_overlays,
                                  add_metadata_overlay)


logger = logging.getLogger(__name__)

# Define artifact function mappings (adjust based on final function names)
SHAPE_ARTIFACT_FUNCS = {
    'edge_ripple': apply_edge_ripple,
    'breaks_holes': apply_breaks_holes,
    'local_elastic': apply_local_elastic,
    'contour_smoothing': apply_contour_smoothing,
    # local_brightness modifies the render, not the mask directly
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
    'gradient_illumination': apply_gradient_illumination,
    'striping_smearing': apply_striping_smearing,
    'fixed_pattern_noise': apply_fixed_pattern_noise, # Note: returns image, noise_map
}

def generate_sample(sample_idx, base_config, master_rng, output_parent_dir):
    """Generates a single synthetic SEM sample with all outputs."""
    start_time = time.time()
    sample_seed = master_rng.randint(0, 2**32 - 1)
    sample_rng = get_rng(sample_seed)
    config = randomize_config_for_sample(base_config, sample_seed)
    logger.info(f"--- Generating Sample {sample_idx:05d} (Seed: {sample_seed}) ---")

    # --- Safely get image settings ---
    image_settings = config.get('image_settings', {}) # Get the sub-dict safely

    h, w = image_settings.get('resolution', [256, 256]) # Default if missing
    bit_depth = image_settings.get('bit_depth', 16) # Default if missing

        # Get magnification (already randomized if range was present)
    magnification = image_settings.get('magnification', 1.0) # Default if missing

    # Get pixel size, providing a default value if missing
    pixel_size_nm_at_1x = image_settings.get('pixel_size_nm_at_1x', None) # Get safely
    if pixel_size_nm_at_1x is None:
        logger.warning("'pixel_size_nm_at_1x' not found in image_settings. Scale bar calculation will be skipped or use default.")
        pixel_size_nm = None # Indicate that pixel size is unknown
    elif magnification == 0: # Avoid division by zero
         logger.warning("Magnification is zero, cannot calculate pixel size. Scale bar may be incorrect.")
         pixel_size_nm = None
    else:
        pixel_size_nm = pixel_size_nm_at_1x / magnification
        logger.debug(f"Calculated pixel size: {pixel_size_nm:.3f} nm")

    out_opts = config['output_options']
    sample_name = f"sem_{sample_idx:05d}"
    sample_output_dir = Path(output_parent_dir) / sample_name
    ensure_dir(sample_output_dir)
    ensure_dir(sample_output_dir / "layers")
    ensure_dir(sample_output_dir / "layers_combined")
    ensure_dir(sample_output_dir / "logs")

    log_file_handler = logging.FileHandler(sample_output_dir / "logs" / "generation.log", mode='w')
    log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    # Avoid adding handlers repeatedly if logger is already configured upstream
    if not logger.handlers:
         logger.addHandler(log_file_handler) # Add only if no handlers exist


    is_negative_control = sample_rng.random() < config.get('run_settings',{}).get('negative_control_probability', 0.0)
    if is_negative_control:
         logger.info("Generating as Negative Control (noise/artifacts only).")
         config['layering']['selected_layers'] = []
         save_text_file("This is a negative control sample.", sample_output_dir / "negative_control_flag.txt")

    raff = Raffler(config.get('artifact_raffle', {}), sample_rng)
    bg_conf = config.get('background', {})
    background_clean = generate_background(bg_conf, (h, w), magnification, sample_rng) # Pass RNG
    initial_background = background_clean.copy()
    image_clean = background_clean.copy()

    all_layers_data = {}
    layer_instance_counts = {}
    layer_masks_original = {}
    layer_masks_actual = {}
    layer_defect_masks = {} # Store defect masks per layer
    layer_renders_actual = [] # List of actual rendered layer buffers (float32)
    instance_id_counter = 1

    selected_layers = config.get('layering', {}).get('selected_layers', [])
    composition_mode = config.get('layering', {}).get('composition_mode', 'additive')
    randomize_layer_order = config.get('layering', {}).get('randomize_order', False)

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
        logger.info(f"Processing Layer {layer_config_idx}: '{layer_id_name}'")

        shape_type = layer_conf['shape']
        pattern_type = layer_conf['pattern']
        intensity = layer_conf['intensity']
        alpha = layer_conf.get('alpha', 1.0)
        shape_params_base = layer_conf['shape_params']
        pattern_params = layer_conf['pattern_params']

        # Get positions using combined params
        positions = get_pattern_positions(pattern_type, (h, w), shape_params_base, pattern_params, sample_rng)

        layer_combined_mask_original = np.zeros((h, w), dtype=np.uint8)
        layer_combined_mask_actual = np.zeros((h, w), dtype=np.uint8)
        layer_render_buffer = np.zeros((h, w), dtype=np.float32) # Float for rendering intensity
        num_instances_in_layer = 0
        applied_shape_artifacts_list = [] # Track artifacts applied in this layer


        for idx, pos in enumerate(positions):
             shape_params = shape_params_base.copy()
             if shape_type.endswith('line') and isinstance(pos, tuple) and len(pos) == 2 and isinstance(pos[0], tuple):
                 shape_params['x1'], shape_params['y1'] = pos[0]
                 shape_params['x2'], shape_params['y2'] = pos[1]
             elif isinstance(pos, tuple) and len(pos) == 2 and isinstance(pos[0], (int, float)):
                 shape_params['center_x'], shape_params['center_y'] = pos
             else:
                 logger.warning(f"Unsupported position format for shape {shape_type}: {pos}")
                 continue

             # --- Generate Original Mask (Per Instance) ---
             # Pass sample_rng for shapes that need it (wavy, polygon)
             original_mask_instance = create_shape_mask(shape_type, shape_params, (h, w), rng=sample_rng)
             layer_combined_mask_original |= original_mask_instance

             # --- Apply Shape-Level Artifacts (Raffled) ---
             actual_mask_instance = original_mask_instance.copy()
             shape_artifacts = raff.raffle_effects('shape') # Raffle PER INSTANCE or per layer? Per layer for now.

             # Apply mask-modifying artifacts
             for artifact in shape_artifacts:
                 if artifact['name'] in SHAPE_ARTIFACT_FUNCS:
                     try:
                         actual_mask_instance = SHAPE_ARTIFACT_FUNCS[artifact['name']](actual_mask_instance, artifact['params'], sample_rng)
                         applied_shape_artifacts_list.append(artifact['name'])
                     except Exception as e:
                         logger.error(f"Error applying shape artifact {artifact['name']} to instance: {e}")
                 elif artifact['name'] == 'local_brightness':
                      # Handled after rendering the shape
                      pass
                 else:
                      logger.warning(f"Shape artifact function '{artifact['name']}' not found or not applicable here.")


             layer_combined_mask_actual |= actual_mask_instance

             # --- Render Actual Shape (Per Instance) ---
             # Use the *actual* mask to render this instance onto the layer buffer
             instance_render = np.zeros_like(layer_render_buffer)
             instance_render[actual_mask_instance > 0] = intensity * alpha # Basic intensity application

             # Apply local brightness artifact if raffled
             brightness_artifact = next((a for a in shape_artifacts if a['name'] == 'local_brightness'), None)
             if brightness_artifact:
                  try:
                       instance_render = apply_local_brightness(instance_render, actual_mask_instance, brightness_artifact['params'], sample_rng)
                       applied_shape_artifacts_list.append('local_brightness')
                  except Exception as e:
                       logger.error(f"Error applying local_brightness artifact: {e}")


             # Add instance render to layer buffer (can change mode later if needed)
             layer_render_buffer += instance_render

             num_instances_in_layer += 1
             instance_id_counter += 1


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


    # 5. Compose Layers
    logger.info(f"Composing {len(layer_renders_actual)} layers using mode: {composition_mode}")
    cumulative_layers_for_gif = [initial_background.copy()]
    for layer_buffer in layer_renders_actual:
         if composition_mode == 'additive': image_clean += layer_buffer
         elif composition_mode == 'multiplicative': image_clean *= (1.0 + layer_buffer * 2) # Example multiplicative blend
         elif composition_mode == 'overwrite':
              # Need the mask for the current layer buffer being applied
              # This requires matching buffer back to layer config - requires careful indexing
              # Let's find the corresponding mask using the render order index
              current_layer_config_idx = layer_indices[len(cumulative_layers_for_gif)-1] # Index of layer config for this buffer
              mask = all_layers_data[current_layer_config_idx]['actual_mask']
              image_clean[mask > 0] = layer_buffer[mask > 0]
         else: image_clean += layer_buffer # Default additive
         image_clean = np.clip(image_clean, 0.0, 1.0) # Clip after each composition
         cumulative_layers_for_gif.append(image_clean.copy())


    image_clean_pre_warp = image_clean.copy()

    # 6. Generate Combined/Instance Masks (Pre-Warp)
    combined_mask_original_pre_warp = generate_combined_mask(layer_masks_original)
    combined_mask_actual_pre_warp = generate_combined_mask(layer_masks_actual)
    instance_mask_pre_warp, instance_meta = generate_instance_mask(layer_masks_actual, layer_instance_counts)


    # 7. Apply Global Artifacts (Geometric, Instrument) - Use Oversized Canvas
    logger.info("Applying global artifacts...")
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
    combined_actual_mask_oversized = embed_in_oversized(combined_mask_actual_pre_warp, oversized_shape, margin_h, margin_w, cv2.BORDER_CONSTANT, 0)
    instance_mask_oversized = embed_in_oversized(instance_mask_pre_warp, oversized_shape, margin_h, margin_w, cv2.BORDER_CONSTANT, 0)

    # Apply Geometric Warps
    geometric_artifacts = raff.raffle_effects('geometric')
    warp_field_combined = None
    masks_to_warp = [m for m in [combined_actual_mask_oversized, instance_mask_oversized] if m is not None]

    for artifact in geometric_artifacts:
        if artifact['name'] in GEOMETRIC_ARTIFACT_FUNCS:
             try:
                  logger.info(f"Applying geometric artifact: {artifact['name']}")
                  image_clean_oversized, warped_masks_out, warp_field = \
                      GEOMETRIC_ARTIFACT_FUNCS[artifact['name']](image_clean_oversized, masks_to_warp, artifact['params'], sample_rng)
                  masks_to_warp = warped_masks_out
                  if warp_field is not None: warp_field_combined = warp_field # Store last warp field
             except Exception as e:
                  logger.error(f"Error applying geometric artifact {artifact['name']}: {e}", exc_info=True)
        else: logger.warning(f"Geometric artifact function '{artifact['name']}' not found.")

    # Update main mask variables after warping
    if masks_to_warp:
         combined_actual_mask_oversized = masks_to_warp[0]
         if len(masks_to_warp) > 1: instance_mask_oversized = masks_to_warp[1]
         else: instance_mask_oversized = None


    # Center Crop Back
    image_clean_warped = image_clean_oversized[margin_h:margin_h+h, margin_w:margin_w+w]
    combined_actual_mask = combined_actual_mask_oversized[margin_h:margin_h+h, margin_w:margin_w+w] if combined_actual_mask_oversized is not None else None
    instance_mask = instance_mask_oversized[margin_h:margin_h+h, margin_w:margin_w+w] if instance_mask_oversized is not None else None
    warp_field_final = warp_field_combined[margin_h:margin_h+h, margin_w:margin_w+w] if warp_field_combined is not None else None


    # Apply Instrument Artifacts
    image_post_instrument = image_clean_warped.copy()
    instrument_artifacts = raff.raffle_effects('instrument')
    applied_instrument_artifacts = []
    total_added_fpn = np.zeros_like(image_post_instrument) # Accumulate FPN separately if needed

    for artifact in instrument_artifacts:
        func = INSTRUMENT_ARTIFACT_FUNCS.get(artifact['name'])
        if func:
            try:
                logger.info(f"Applying instrument artifact: {artifact['name']}")
                # Special handling for functions needing extra context or returning noise maps
                if artifact['name'] == 'topographic_shading':
                     image_post_instrument = func(image_post_instrument, all_layers_data, artifact['params'], sample_rng)
                elif artifact['name'] == 'fixed_pattern_noise':
                     image_post_instrument, fpn_map = func(image_post_instrument, artifact['params'], sample_rng)
                     total_added_fpn += fpn_map # Accumulate FPN if needed separately
                else:
                     image_post_instrument = func(image_post_instrument, artifact['params'], sample_rng)

                applied_instrument_artifacts.append(artifact['name'])
            except Exception as e:
                logger.error(f"Error applying instrument artifact {artifact['name']}: {e}", exc_info=True)
        else:
             logger.warning(f"Instrument artifact function '{artifact['name']}' not found.")

    image_post_instrument = np.clip(image_post_instrument, 0.0, 1.0)


    # 8. Apply Detector Noise
    logger.info("Applying detector noise...")
    image_final_noisy = image_post_instrument.copy()
    # Start with FPN map if generated, otherwise zeros
    total_added_noise = total_added_fpn.copy()
    noise_artifacts = raff.raffle_effects('noise')
    applied_noise_artifacts = []

    # Apply quantization first if present
    quant_artifact = next((a for a in noise_artifacts if a['name'] == 'quantization'), None)
    if quant_artifact:
         try:
              logger.info(f"Applying noise: {quant_artifact['name']}")
              image_final_noisy, q_noise = apply_noise(image_final_noisy, quant_artifact['name'], quant_artifact['params'], sample_rng)
              total_added_noise += q_noise
              applied_noise_artifacts.append(quant_artifact['name'])
         except Exception as e: logger.error(f"Error applying quantization: {e}", exc_info=True)

    # Apply other noise types
    for artifact in noise_artifacts:
        if artifact['name'] == 'quantization': continue # Already applied
        noise_type = artifact['name']
        try:
             logger.info(f"Applying noise: {noise_type}")
             image_final_noisy, added_noise = apply_noise(image_final_noisy, noise_type, artifact['params'], sample_rng)
             if added_noise is not None: total_added_noise += added_noise
             applied_noise_artifacts.append(noise_type)
        except Exception as e: logger.error(f"Error applying noise {noise_type}: {e}", exc_info=True)

    image_final_noisy = np.clip(image_final_noisy, 0.0, 1.0)


    # 9. Generate Overlays and Final Visualizations
    logger.info("Generating overlays and visualizations...")
    final_image_vis_8bit = image_to_bit_depth(image_final_noisy, 8) # Use 8-bit for overlays

    overlay_contour_vis, instance_mask_vis, warp_field_vis = create_overlays(
        final_image_vis_8bit,
        combined_actual_mask,
        instance_mask,
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
        "pixel_size_nm_at_1x": pixel_size_nm_at_1x, # Store the base value used
        "pixel_size_nm_calculated": pixel_size_nm, # Store the calculated value (could be None)
        "background_type": config.get('background',{}).get('selected_type'),
        "layers": [{'config_idx': idx, 'layer_id': data['id']} for idx, data in all_layers_data.items()],
        "num_layers": len(selected_layers), "composition_mode": composition_mode,
        "applied_shape_artifacts": list(set(a for data in all_layers_data.values() for a in data['applied_shape_artifacts'])),
        "applied_geometric_artifacts": [a['name'] for a in geometric_artifacts],
        "applied_instrument_artifacts": applied_instrument_artifacts,
        "applied_noise_artifacts": applied_noise_artifacts,
        "instance_info": instance_meta,
        "generation_time_sec": round(time.time() - start_time, 2),
        "output_paths": {}
    }


    # 11. Save Outputs
    logger.info("Saving outputs...")
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
                        logger.warning("Max instance ID > 65535, saving as 16-bit PNG might lose IDs. Consider TIF or NPY.")
                        img_to_save = instance_mask.astype(np.uint16)
                    else:
                        img_to_save = instance_mask.astype(np.uint16 if np.max(instance_mask)>255 else np.uint8)
                    save_image_data(img_to_save, path, bit_depth=16 if img_to_save.dtype==np.uint16 else 8, format_hint='PNG')
                    saved_successfully = path.is_file()
                else:
                    logger.warning(f"Unsupported instance mask format '{inst_mask_format}'. Saving as NPY.")

            except Exception as e:
                logger.error(f"Error saving instance mask as {inst_mask_format}: {e}", exc_info=True)

            # Fallback to NPY if specified format failed or wasn't image format
            if not saved_successfully:
                logger.warning(f"Failed to save instance mask as {inst_mask_format}. Saving as NPY.")
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
    logger.removeHandler(log_file_handler)
    log_file_handler.close()

    logger.info(f"--- Finished Sample {sample_idx:05d} in {time.time() - start_time:.2f} seconds ---")

    return True # Indicate success
