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
        logger.error(f"Failed to save NumPy array {path}: {e}")

def save_image_data(image_data, path, bit_depth=16, format_hint=None):
     """Saves image data (float 0-1) to file (TIF/PNG)."""
     ensure_dir(Path(path).parent)
     # Determine format from extension or hint
     file_ext = Path(path).suffix.lower()
     fmt = format_hint
     if not fmt:
         if file_ext == '.tif' or file_ext == '.tiff':
             fmt = 'TIFF'
         elif file_ext == '.png':
             fmt = 'PNG'
         else:
             logger.warning(f"Cannot determine image format for {path}, defaulting to PNG.")
             fmt = 'PNG'
             path = Path(path).with_suffix('.png') # Ensure correct extension

     try:
         # Convert normalized float (0-1) to target bit depth
         image_typed = image_to_bit_depth(image_data, bit_depth)

         if fmt == 'TIFF':
              # Use imageio with FreeImage plugin for better TIFF support
              imageio.imwrite(path, image_typed, format='TIFF-FI')
         elif fmt == 'PNG':
              # PNG typically uint8 or uint16
              if bit_depth not in [8, 16]:
                  logger.warning(f"PNG format requested for bit depth {bit_depth}. Saving as 16-bit PNG.")
                  image_typed = image_to_bit_depth(image_data, 16)
              imageio.imwrite(path, image_typed, format='PNG-FI') # Use FreeImage plugin if available
         else:
             logger.error(f"Unsupported image format hint: {fmt}")
             return

         logger.debug(f"Saved {bit_depth}-bit image ({fmt}): {path}")

     except Exception as e:
         logger.error(f"Failed to save image {path}: {e}")


def save_json_data(data, path):
    ensure_dir(Path(path).parent)
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.debug(f"Saved JSON: {path}")
    except Exception as e:
        logger.error(f"Failed to save JSON {path}: {e}")

def save_gif_data(frames, path, duration=0.1):
     """Saves a list of frames (uint8) as an animated GIF."""
     ensure_dir(Path(path).parent)
     # Ensure frames are uint8
     uint8_frames = []
     for frame in frames:
         if frame.dtype != np.uint8:
             # Assume float 0-1, convert to uint8
             if np.max(frame) <= 1.0 and np.min(frame) >= 0.0:
                  uint8_frames.append((frame * 255).astype(np.uint8))
             else: # Assume already scaled 0-255 but wrong type
                 uint8_frames.append(frame.astype(np.uint8))
         else:
             uint8_frames.append(frame)

     try:
         # imageio handles duration in seconds per frame
         frame_duration_sec = duration if isinstance(duration, (int, float)) else 0.1 # Default if invalid
         imageio.mimsave(path, uint8_frames, duration=frame_duration_sec, subrectangles=True) # Use subrectangles for potential optimization
         logger.debug(f"Saved GIF: {path}")
     except Exception as e:
         logger.error(f"Failed to save GIF {path}: {e}")


def save_text_file(content, path):
    ensure_dir(Path(path).parent)
    try:
        with open(path, 'w') as f:
            f.write(content)
        logger.debug(f"Saved text file: {path}")
    except Exception as e:
        logger.error(f"Failed to save text file {path}: {e}")

def calculate_hashes(file_paths):
    """Calculates SHA256 hashes for a list of files."""
    hashes = {}
    for file_path in file_paths:
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    hasher.update(chunk)
            hashes[os.path.basename(file_path)] = hasher.hexdigest()
        except FileNotFoundError:
            logger.warning(f"File not found for hashing: {file_path}")
            hashes[os.path.basename(file_path)] = None
        except Exception as e:
             logger.error(f"Error hashing file {file_path}: {e}")
             hashes[os.path.basename(file_path)] = None
    return hashes
