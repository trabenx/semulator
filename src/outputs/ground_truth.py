import numpy as np
import cv2
from ..core.utils import create_color_visualization, get_distinct_colors, visualize_warp_field
import logging

logger = logging.getLogger(__name__)

def generate_instance_data(warped_semantic_map=None, layer_masks_actual=None):
     """
     Generates a final instance mask and extracts instance metadata primarily
     from a warped semantic map (layer index map). Can optionally use pre-warped
     layer masks as a fallback (currently less preferred).

     Args:
         warped_semantic_map (np.ndarray | None): Warped map where pixel value is
             layer_config_idx + 1 (0 for background). Shape (H, W), dtype uint8.
         layer_masks_actual (dict | None): Deprecated - primarily use semantic map.
             {layer_idx: actual_mask_for_layer (HxW, uint8)} - represents masks *before* warping.

     Returns:
         tuple: (
             instance_mask (np.ndarray|None): Final instance mask after warp
                 (HxW, uint16/uint32), or None if generation failed.
             instance_metadata (dict): {instance_id: {'layer_config_idx': L,
                 'bbox_xywh': [x,y,w,h], 'centroid_xy': [cx, cy]}}
                 Coordinates are relative to the final warped image frame.
         )
     """
     instance_mask = None
     instance_metadata = {}

     if warped_semantic_map is None or not isinstance(warped_semantic_map, np.ndarray):
         logger.warning("Cannot generate instance data: Warped semantic map is missing or invalid.")
         # Optional: Fallback to using layer_masks_actual if needed, but that requires
         # warping them individually which is complex and not implemented here.
         return None, {}

     h, w = warped_semantic_map.shape
     logger.info("Generating instance data from warped semantic map...")

     try:
         # Binarize the semantic map (any layer > 0 is foreground)
         binary_map_for_cc = (warped_semantic_map > 0).astype(np.uint8)

         # Find connected components in the warped map
         num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
             binary_map_for_cc, connectivity=8, ltype=cv2.CV_32S
         )

         max_instance_id = num_labels - 1
         if max_instance_id <= 0:
             logger.warning("No instances found in warped semantic map.")
             return np.zeros((h, w), dtype=np.uint16), {} # Return empty map

         # Determine required dtype for final instance mask
         dtype = np.uint16 if max_instance_id < 65535 else np.uint32
         if max_instance_id >= 2**32:
              logger.warning(f"Exceeded maximum instance ID limit for uint32 ({max_instance_id})!")
              # Continue but some IDs might wrap around if not handled downstream

         # Create the final instance mask (labels start from 1)
         instance_mask = labels_im.astype(dtype)

         # Extract metadata (bbox, centroid, layer_idx)
         # stats columns: 0:left(x), 1:top(y), 2:width, 3:height, 4:area
         # centroids columns: 0:x, 1:y
         for inst_id in range(1, num_labels): # Skip background label 0
             # Clip instance ID if it exceeds dtype max (shouldn't happen if check above works)
             current_inst_id_clipped = min(inst_id, np.iinfo(dtype).max)
             if current_inst_id_clipped != inst_id:
                  logger.warning(f"Clipping instance ID {inst_id} to {current_inst_id_clipped}")

             # Get stats (ensure indexing is correct)
             x, y, w_box, h_box, area = stats[inst_id]
             cx, cy = centroids[inst_id]

             # Determine original layer index from semantic map at centroid
             int_cx, int_cy = int(round(cx)), int(round(cy))
             layer_config_idx = -1 # Default if unknown
             if 0 <= int_cy < h and 0 <= int_cx < w: # Check bounds
                 semantic_value = warped_semantic_map[int_cy, int_cx]
                 if semantic_value > 0:
                     layer_config_idx = int(semantic_value) - 1 # Map stores index + 1
                 else:
                      # Centroid landed on background in semantic map (e.g., due to warping near edge)
                      # Try sampling near centroid or use majority vote in bbox? For now, mark as unknown.
                      logger.debug(f"Centroid for instance {inst_id} landed on background (value 0) in semantic map.")
             else:
                  logger.debug(f"Centroid for instance {inst_id} ({cx:.1f}, {cy:.1f}) is outside image bounds.")


             instance_metadata[current_inst_id_clipped] = {
                 'layer_config_idx': layer_config_idx, # The original index from the config list
                 'bbox_xywh': [int(x), int(y), int(w_box), int(h_box)],
                 'centroid_xy': [float(cx), float(cy)]
             }

         logger.info(f"Generated instance data with {max_instance_id} instances.")

     except Exception as e:
          logger.error(f"Error during instance data generation from semantic map: {e}", exc_info=True)
          return None, {} # Return None on failure

     return instance_mask, instance_metadata



def generate_combined_mask(layer_masks, mask_type='actual'):
    """Combines per-layer masks into a single binary mask."""
    if not layer_masks:
        return None

    first_mask = next(iter(layer_masks.values()))
    h, w = first_mask.shape
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for layer_idx in sorted(layer_masks.keys()):
        mask = layer_masks[layer_idx]
        if mask is not None:
             combined_mask[mask > 0] = 1 # Combine using logical OR

    return combined_mask

def generate_defect_mask(original_mask, actual_mask):
     """Calculates the XOR difference between original and actual masks."""
     if original_mask is None or actual_mask is None or original_mask.shape != actual_mask.shape:
         return None
     # Ensure binary (0/1) before XOR
     orig_bin = (original_mask > 0).astype(np.uint8)
     act_bin = (actual_mask > 0).astype(np.uint8)
     defect_mask = cv2.bitwise_xor(orig_bin, act_bin)
     return defect_mask

def create_overlays(image_vis, combined_mask, instance_mask, warp_field, layer_id_to_color):
     """Generates debug overlays (contours, instance vis, warp vis)."""
     # Example: Draw contours of combined mask on image
     overlay_contour_vis = image_vis.copy()
     if overlay_contour_vis.ndim == 2:  # Convert grayscale to BGR for color drawing
         overlay_contour_vis = cv2.cvtColor(overlay_contour_vis, cv2.COLOR_GRAY2BGR)

     # Combined mask contours
     if combined_mask is not None:
         contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
         cv2.drawContours(overlay_contour_vis, contours, -1, (0, 255, 0), 1) # Green contours

     # Instance mask visualization
     instance_mask_vis = None
     if instance_mask is not None:
          max_id = np.max(instance_mask)
          # Create a colormap based on instance IDs
          # Note: get_distinct_colors might be slow for >> 1000s of instances
          # A simpler hash-based or repeating colormap might be needed for large counts
          if max_id > 0:
              if max_id < 1000: # Use distinct colors for fewer instances
                  colors = get_distinct_colors(max_id) # Generate max_id colors
                  # Map ID i to colors[i-1] (since IDs start at 1)
                  inst_colormap = {i: colors[(i-1) % len(colors)] for i in range(1, max_id + 1)}  # Map IDs to colors
              else: # Fallback for many instances (e.g., cyclical map)
                  base_colors = get_distinct_colors(20) # Use a smaller set and repeat
                  inst_colormap = {i: base_colors[(i-1) % len(base_colors)] for i in range(1, max_id + 1)}
              instance_mask_vis = create_color_visualization(instance_mask, inst_colormap)


     # Warp field visualization
     warp_field_vis = None
     if warp_field is not None:
          warp_field_vis = visualize_warp_field(warp_field)


     return overlay_contour_vis, instance_mask_vis, warp_field_vis


def add_metadata_overlay(image, scale_bar=True, text=None, pixel_size_nm=None):
    """Adds scale bars and text to an image. Uses calculated pixel_size_nm."""
    output_image = image.copy()
    if output_image.ndim == 2:
         output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
    h, w = output_image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (255, 255, 255) # White

    if scale_bar and pixel_size_nm and pixel_size_nm > 1e-6:
        # Determine appropriate scale bar length (e.g., 10nm, 50nm, 100nm, 0.5um, 1um)
        target_lengths_nm = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        ideal_bar_width_px = w * 0.15 # Aim for bar around 15% of image width

        best_length_nm = target_lengths_nm[0]
        min_diff = float('inf')

        for length_nm in target_lengths_nm:
            bar_width_px = length_nm / pixel_size_nm
            diff = abs(bar_width_px - ideal_bar_width_px)
            # Ensure bar is reasonably large but not too large
            if diff < min_diff and 10 < bar_width_px < w * 0.4:
                min_diff = diff
                best_length_nm = length_nm

        bar_width_px = int(round(best_length_nm / pixel_size_nm))
        # Format scale bar text
        if best_length_nm >= 1000:
             bar_text = f"{best_length_nm / 1000:.1f}".rstrip('0').rstrip('.') + " um"
        else:
             bar_text = f"{best_length_nm:.0f} nm"

        # Draw the bar bottom right
        bar_h_px = max(1, h // 80) # Bar thickness
        margin_x = int(w * 0.03)
        margin_y = int(h * 0.03)
        start_x = w - bar_width_px - margin_x
        start_y = h - bar_h_px - margin_y
        end_x = start_x + bar_width_px
        end_y = start_y + bar_h_px
        cv2.rectangle(output_image, (start_x, start_y), (end_x, end_y), font_color, -1)

        # Draw text below bar
        font_scale = 0.4
        font_thickness = 1
        text_size, _ = cv2.getTextSize(bar_text, font, font_scale, font_thickness)
        text_x = start_x + (bar_width_px - text_size[0]) // 2
        text_y = start_y - text_size[1] // 2 # Position text above the bar
        cv2.putText(output_image, bar_text, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    if text:
        # Draw text top left
        font_scale = 0.5
        font_thickness = 1
        text_y = int(h * 0.05)
        for i, line in enumerate(text.split('\n')):
             line_y = text_y + i * int(h*0.04)
             cv2.putText(output_image, line, (int(w * 0.02), line_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    return output_image
