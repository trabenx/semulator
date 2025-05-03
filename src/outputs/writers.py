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
     """Saves image data (float 0-1) to file (TIF/PNG)."""
     # Ensure path is a Path object
     path = Path(path)
     ensure_dir(path.parent)

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
             path = path.with_suffix('.png') # Ensure correct extension

     try:
        # Convert normalized float (0-1) to target bit depth
        # Ensure input is numpy array before conversion
        if not isinstance(image_data, np.ndarray):
             image_data = np.array(image_data)

        image_typed = image_to_bit_depth(image_data, bit_depth)

        if fmt == 'TIFF':
            try:
                # Try tifffile explicitly first
                imageio.v3.imwrite(path, image_typed, plugin='tifffile')
                logger.debug(f"Saved {bit_depth}-bit image (TIFF via tifffile): {path}")
            except Exception as e_tf:
                logger.warning(f"Saving TIFF with tifffile failed ({e_tf}), trying default...")
                # Fallback to default imageio handling if tifffile fails
                imageio.imwrite(path, image_typed, format='TIFF')
                logger.debug(f"Saved {bit_depth}-bit image (TIFF via default): {path}")
        elif fmt == 'PNG':
            imageio.imwrite(path, image_typed, format='PNG')
            logger.debug(f"Saved {bit_depth}-bit image (PNG via default): {path}")

     except FileNotFoundError:
          logger.error(f"Failed to save image {path}: File path seems invalid or inaccessible.")
     except ImportError:
          logger.error(f"Failed to save image {path}: imageio backend for {fmt} might be missing.")
     except Exception as e:
          logger.error(f"Failed to save image {path}: {e}", exc_info=True) # Add exc_info


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
