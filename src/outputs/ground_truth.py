import numpy as np
import cv2
from ..core.utils import create_color_visualization, get_distinct_colors, visualize_warp_field
import logging

logger = logging.getLogger(__name__)

def generate_instance_mask(layer_masks, layer_instance_counts):
     """
     Generates a combined instance mask from per-layer masks.
     Assigns unique IDs starting from 1.
     Args:
         layer_masks (dict): {layer_idx: actual_mask_for_layer (HxW, uint8)}
         layer_instance_counts (dict): {layer_idx: num_instances_in_layer}
     Returns:
         np.ndarray: Instance mask (HxW, uint16 or uint32)
         dict: Mappings {instance_id: {'layer_idx': L, 'instance_in_layer': I}}
     """
     if not layer_masks:
         return None, {}

     # Determine required dtype (uint16 max ~65k instances)
     total_instances = sum(layer_instance_counts.values())
     dtype = np.uint16 if total_instances < 65535 else np.uint32
     if total_instances >= 2**32:
          logger.warning("Exceeded maximum number of instances for uint32 mask!")
          # Handle error or clamp? Clamp for now.
          dtype = np.uint32


     first_mask = next(iter(layer_masks.values()))
     h, w = first_mask.shape
     instance_mask = np.zeros((h, w), dtype=dtype)
     instance_metadata = {}
     current_instance_id = 1

     # Sort layers by index for consistent ID assignment
     sorted_layer_indices = sorted(layer_masks.keys())

     for layer_idx in sorted_layer_indices:
         layer_mask = layer_masks[layer_idx]
         num_instances = layer_instance_counts[layer_idx]

         # Find connected components in the layer mask
         # connectivity=8 means pixels are connected if they touch at edges or corners
         num_labels, labels_im = cv2.connectedComponents(layer_mask, connectivity=8)

         # num_labels includes the background (0), so iterate from 1 up to num_labels-1
         instance_in_layer_idx = 0
         for label_id in range(1, num_labels):
              if current_instance_id > np.iinfo(dtype).max:
                  logger.error("Instance ID exceeds dtype limit. Stopping assignment.")
                  break

              component_mask = (labels_im == label_id)
              # Assign unique ID, only where not already assigned by previous layers
              instance_mask[component_mask & (instance_mask == 0)] = current_instance_id

              instance_metadata[current_instance_id] = {
                  'layer_idx': layer_idx,
                  'instance_in_layer': instance_in_layer_idx
              }
              current_instance_id += 1
              instance_in_layer_idx += 1

         if current_instance_id > np.iinfo(dtype).max: break # Break outer loop too

     logger.info(f"Generated instance mask with {current_instance_id - 1} instances.")
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
