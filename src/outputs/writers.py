import numpy as np
import imageio
import json
import hashlib
import logging
import os
from pathlib import Path
from ..core.utils import ensure_dir, image_to_bit_depth, normalize_image

logger = logging.getLogger(__name__)

def save_numpy(array, path):
    ensure_dir(Path(path).parent)
    try:
        np.save(path, array)
        logger.debug(f"Saved NumPy array: {path}")
    except Exception as e:
        logger.error(f"Failed to save NumPy array {path}: {e}", exc_info=True) # Add exc_info

def save_image_data(image_data, path, bit_depth=16, format_hint=None):
     """Saves image data (handles float or integer) to file (TIF/PNG)."""
     # Ensure path is a Path object
     path = Path(path)
     ensure_dir(path.parent) # Ensure directory first

     # --- Input Data Checks ---
     if image_data is None:
         logger.error(f"Cannot save image to {path}: Input data is None.")
         return False # Indicate failure
     if not isinstance(image_data, np.ndarray):
         try: image_data = np.array(image_data)
         except Exception as e_conv: logger.error(f"Failed to convert input to NumPy array for {path}: {e_conv}"); return False
     if not np.all(np.isfinite(image_data)):
          logger.warning(f"Non-finite values (NaN/inf) found in image data for {path}. Clamping.")
          image_data = np.nan_to_num(image_data, nan=0.0, posinf=1.0, neginf=0.0) # Clamp/replace bad values
     # ---

     # Determine format from extension or hint
     file_ext = path.suffix.lower()
     fmt = format_hint
     if not fmt:
         if file_ext in ['.tif', '.tiff']:
             fmt = 'TIFF'
         elif file_ext == '.png':
             fmt = 'PNG'
         else:
             logger.warning(f"Cannot determine image format for {path}, defaulting to PNG.")
             fmt = 'PNG'
         path = path.with_suffix(f'.{fmt.lower()}')

     try:
         image_typed = None
         save_dtype_log = "" # For logging

         # --- Prepare data for saving ---
         if np.issubdtype(image_data.dtype, np.integer):
              # Integer data (e.g., masks) - save directly
              image_typed = image_data
              save_dtype_log = f"integer data as is ({image_typed.dtype})"
              # Infer bit depth for logging consistency
              if image_typed.dtype == np.uint8: bit_depth = 8
              elif image_typed.dtype == np.uint16: bit_depth = 16
              elif image_typed.dtype == np.uint32: bit_depth = 32
         elif np.issubdtype(image_data.dtype, np.floating):
              # Float data - normalize [0,1] then convert to target bit depth
              save_dtype_log = f"float data to {bit_depth}-bit"
              # Ensure data is clipped 0-1 before conversion
              image_clipped = np.clip(image_data, 0.0, 1.0)
              image_typed = image_to_bit_depth(image_clipped, bit_depth)
         else:
              # Other types - attempt conversion? Log warning.
              logger.warning(f"Unexpected image data type {image_data.dtype} for {path}. Attempting conversion to {bit_depth}-bit.")
              try: # Try normalizing assuming it's numeric-like
                  image_normalized = normalize_image(image_data.astype(float))
                  image_typed = image_to_bit_depth(image_normalized, bit_depth)
                  save_dtype_log = f"unexpected type to {bit_depth}-bit"
              except Exception as e_conv_other:
                   logger.error(f"Cannot convert dtype {image_data.dtype} for saving {path}: {e_conv_other}")
                   return False # Indicate failure

         if image_typed is None:
             logger.error(f"Failed to prepare typed image data for saving {path}.")
             return False

         # --- Save using simple imageio call ---
         logger.debug(f"Saving {save_dtype_log} to {path} (Format: {fmt}, Dtype: {image_typed.dtype})")
         imageio.imwrite(path, image_typed, format=fmt)
         logger.debug(f"imageio.imwrite call completed for {path}.")

         # --- Verify Save ---
         # Short delay might help filesystem cache flushing on some systems
         import time
         time.sleep(0.05)
         if not path.is_file():
             logger.error(f"File verification FAILED after saving {path}. Check permissions/disk space.")
             return False # Indicate failure

         logger.debug(f"Successfully saved {path}")
         return True # Indicate success

     except Exception as e:
          logger.error(f"!!! EXCEPTION during save_image_data for {path}: {e}", exc_info=True)
          return False # Indicate failure


def save_json_data(data, path):
    ensure_dir(Path(path).parent)
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.debug(f"Saved JSON: {path}")
    except Exception as e:
        logger.error(f"Failed to save JSON {path}: {e}", exc_info=True) # Add exc_info

def save_gif_data(frames, path, duration=0.1):
     """Saves a list of frames (uint8) as an animated GIF."""
     path = Path(path)
     ensure_dir(path.parent)
     uint8_frames = []
     try:
         for i, frame in enumerate(frames):
             if not isinstance(frame, np.ndarray): frame = np.array(frame) # Ensure numpy array
             if frame.dtype != np.uint8:
                 if np.max(frame) <= 1.0 and np.min(frame) >= 0.0: # Assume float 0-1
                      uint8_frames.append((np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8))
                 elif np.max(frame) <= 255 and np.min(frame) >= 0: # Assume scaled 0-255 but wrong type
                     uint8_frames.append(frame.astype(np.uint8))
                 else: # Unknown range, attempt normalization
                      logger.warning(f"Frame {i} for GIF {path} has unknown range/type ({frame.dtype}, min={np.min(frame)}, max={np.max(frame)}). Attempting normalization.")
                      norm_frame = normalize_image(frame.astype(float)) # Normalize to 0-1
                      uint8_frames.append((norm_frame * 255).astype(np.uint8))

             else: # Already uint8
                 uint8_frames.append(frame)

         frame_duration_sec = duration if isinstance(duration, (int, float)) else 0.1
         imageio.mimsave(path, uint8_frames, duration=frame_duration_sec, subrectangles=True)
         logger.debug(f"Saved GIF: {path}")
     except Exception as e:
         logger.error(f"Failed to save GIF {path}: {e}", exc_info=True) # Add exc_info


def save_text_file(content, path):
    ensure_dir(Path(path).parent)
    try:
        with open(path, 'w') as f:
            f.write(content)
        logger.debug(f"Saved text file: {path}")
    except Exception as e:
        logger.error(f"Failed to save text file {path}: {e}", exc_info=True) # Add exc_info

def calculate_hashes(file_paths):
    """Calculates SHA256 hashes for a list of files."""
    hashes = {}
    for file_path in file_paths:
        try:
            if not Path(file_path).is_file():
                 logger.warning(f"File not found for hashing (skipping): {file_path}")
                 continue
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(4096)
                    if not chunk: break
                    hasher.update(chunk)
            hashes[os.path.basename(file_path)] = hasher.hexdigest()
        except Exception as e:
             logger.error(f"Error hashing file {file_path}: {e}", exc_info=True) # Add exc_info
             hashes[os.path.basename(file_path)] = None
    return hashes
