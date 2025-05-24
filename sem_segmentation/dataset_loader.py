# sem_segmentation/dataset_loader.py
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset
import imageio.v3 as iio

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src' / 'core')) # Add src/core to path
from constants import SHAPE_TYPE_MAP

import logging

logger = logging.getLogger(__name__)

# You'll need MAX_PREDICTABLE_LAYERS here
# One way: from ..src.core.constants import MAX_PREDICTABLE_LAYERS
# Or pass it as an argument to the dataset constructor
# For simplicity, let's assume it's passed as an argument: max_layers_to_load

class SEMDataset(Dataset):
    def __init__(self, data_dir, target_size=(512, 512), augment=False, use_tif=False,
                 num_classes=None, max_layers_to_load=5): # Added max_layers_to_load
        self.data_dir = Path(data_dir)
        self.target_size = list(target_size)
        self.augment = augment
        self.use_tif = use_tif
        self.num_classes = num_classes # Total number of shape types + background
        self.max_layers_to_load = max_layers_to_load # Max layer "slots"

        self.sample_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('sem_')])
        if not self.sample_dirs:
            raise FileNotFoundError(f"No sample directories found in {data_dir}")
        logger.info(f"Found {len(self.sample_dirs)} samples in {data_dir}. Max layers to load: {self.max_layers_to_load}")

    def __len__(self):
        return len(self.sample_dirs)


    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        if self.use_tif:
            # Try to find the primary .tif image directly in the parent output dir
            # e.g., generated_dataset/sem_00001.tif
            img_name_base = sample_dir.name
            img_path = self.data_dir / f"{img_name_base}.tif"
            if not img_path.is_file(): # Fallback to looking inside the sample dir
                img_path = sample_dir / f"{img_name_base}.tif"
        else:
            img_path = sample_dir / "image_final_noisy_vis.png"

        # Load image
        try:
            if self.use_tif:
                image = iio.imread(img_path) # Reads TIF as numpy array
                if image.ndim == 2: # Grayscale
                    image = image[:, :, np.newaxis] # Add channel dimension -> H, W, C
                # Normalize 16-bit or 8-bit TIF to float32 [0, 1]
                if image.dtype == np.uint16:
                    image = image.astype(np.float32) / 65535.0
                elif image.dtype == np.uint8:
                    image = image.astype(np.float32) / 255.0
                else: # Other types, attempt normalization
                    min_val, max_val = np.min(image), np.max(image)
                    if max_val > min_val:
                         image = (image.astype(np.float32) - min_val) / (max_val - min_val)
                    else:
                         image = np.zeros_like(image, dtype=np.float32)

            else: # PNG
                image = Image.open(img_path).convert('L') # Grayscale Pillow image
                image = np.array(image, dtype=np.float32) / 255.0 # H, W
                image = image[:, :, np.newaxis] # Add channel dimension -> H, W, C
        except Exception as e:
            raise IOError(f"Error loading image {img_path}: {e}")


        # --- Load Semantic Mask for Shape Types ---
        mask_path = sample_dir / "shape_type_semantic_mask.npy"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Semantic mask file not found: {mask_path}")
        try:
            mask = np.load(mask_path).astype(np.int64) # H, W, with class IDs
        except Exception as e:
            raise IOError(f"Error loading mask {mask_path}: {e}")

        # --- Convert to PIL for some augmentations ---
        # Image: from H,W,C (float32 [0,1]) to PIL 'L'
        image_pil = Image.fromarray((image.squeeze() * 255).astype(np.uint8), mode='L')

        # --- Load Per-Layer Semantic Masks ---
        # Initialize a tensor to hold all layer masks
        # Shape: (max_layers_to_load, OriginalH, OriginalW)
        # We'll resize after loading all and potentially augmenting PIL versions
        # Load into a list of numpy arrays first
        loaded_layer_masks_np = []
        original_mask_shape = None # Store shape from first valid mask
        # Determine original H, W from image first, if possible
        if image is not None:
            original_mask_shape = (image.shape[0], image.shape[1])
        else: # Fallback if image loading failed, less ideal
            original_mask_shape = self.target_size # This might be problematic if target_size isn't original

        for i in range(self.max_layers_to_load):
            # Try to find the layer's specific semantic mask
            # Path construction depends on how generator saves them (e.g., in the layer_XX subdir)
            mask_file_name = f"shape_type_semantic_mask.npy" # Name used in generator proposal
            mask_path = sample_dir / "layers" / f"layer_{i:02d}" / mask_file_name

            if mask_path.is_file():
                try:
                    layer_mask_np = np.load(mask_path).astype(np.int64)
                    if original_mask_shape is None: original_mask_shape = layer_mask_np.shape
                    # Ensure loaded mask has expected shape before appending
                    if layer_mask_np.shape != original_mask_shape:
                         logger.warning(f"Mask {mask_path} shape {layer_mask_np.shape} mismatch, expected {original_mask_shape}. Resizing/Padding.")
                         # Simplistic resize/pad - may need more robust handling
                         temp_pil = Image.fromarray(layer_mask_np.astype(np.uint8), mode='L')
                         temp_pil_resized = TF.resize(temp_pil, list(original_mask_shape), interpolation=TF.InterpolationMode.NEAREST)
                         layer_mask_np = np.array(temp_pil_resized, dtype=np.int64)
                    loaded_layer_masks_np.append(layer_mask_np)
                except Exception as e:
                    logger.warning(f"Error loading layer mask {mask_path} for sample {sample_dir.name}: {e}. Using empty mask.")
                    loaded_layer_masks_np.append(np.zeros(original_mask_shape if original_mask_shape else self.target_size, dtype=np.int64))
            else:
                loaded_layer_masks_np.append(np.zeros(original_mask_shape if original_mask_shape else self.target_size, dtype=np.int64))


        if not original_mask_shape and image is not None: # Fallback if no masks loaded but image did
            original_mask_shape = (image.shape[0], image.shape[1])

        # Ensure all placeholder masks have the correct original_mask_shape if it was determined late
        for i in range(len(loaded_layer_masks_np)):
            if loaded_layer_masks_np[i].shape != original_mask_shape and original_mask_shape is not None:
                logger.warning(f"Correcting shape of placeholder mask for layer {i}")
                loaded_layer_masks_np[i] = np.zeros(original_mask_shape, dtype=np.int64)


        # Convert list of numpy masks to list of PIL masks for augmentation
        layer_masks_pil = [Image.fromarray(m.astype(np.uint8), mode='L') for m in loaded_layer_masks_np]

        # --- Augmentation (applied to PIL Images) ---
        if self.augment:
            # Apply the *same* geometric augmentation to the image and *all* layer masks
            if random.random() > 0.5: # Horizontal Flip
                image_pil = TF.hflip(image_pil)
                layer_masks_pil = [TF.hflip(m) for m in layer_masks_pil]
            if random.random() > 0.5: # Vertical Flip
                image_pil = TF.vflip(image_pil)
                layer_masks_pil = [TF.vflip(m) for m in layer_masks_pil]

            angle = random.choice([0, 0, 0, 90, 180, 270]) # More chance for 0 rotation
            if angle != 0:
                image_pil = TF.rotate(image_pil, angle, interpolation=TF.InterpolationMode.BILINEAR)
                layer_masks_pil = [TF.rotate(m, angle, interpolation=TF.InterpolationMode.NEAREST) for m in layer_masks_pil]

            # Color Jitter (Brightness, Contrast) - for image only
            if random.random() > 0.3: # Apply with 30% chance
                brightness_factor = random.uniform(0.7, 1.3)
                image_pil = ImageEnhance.Brightness(image_pil).enhance(brightness_factor)
            if random.random() > 0.3:
                contrast_factor = random.uniform(0.7, 1.3)
                image_pil = ImageEnhance.Contrast(image_pil).enhance(contrast_factor)

            # Gaussian Blur - for image only
            if random.random() > 0.2: # Apply with 20% chance
                blur_radius = random.uniform(0.1, 1.5)
                image_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            # Small Affine (apply to image and all masks)
            if random.random() > 0.2:
                affine_angle = random.uniform(-7, 7); max_translate = 0.07 * self.target_size[0]
                translate_x = random.uniform(-max_translate, max_translate); translate_y = random.uniform(-max_translate, max_translate)
                scale = random.uniform(0.93, 1.07); shear = random.uniform(-3, 3)
                image_pil = TF.affine(image_pil, angle=affine_angle, translate=(translate_x, translate_y), scale=scale, shear=shear, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
                layer_masks_pil = [TF.affine(m, angle=affine_angle, translate=(translate_x, translate_y), scale=scale, shear=shear, interpolation=TF.InterpolationMode.NEAREST, fill=0) for m in layer_masks_pil]


        # --- Resize (after augmentations) ---
        image_pil_resized = TF.resize(image_pil, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)
        layer_masks_pil_resized = [TF.resize(m, self.target_size, interpolation=TF.InterpolationMode.NEAREST) for m in layer_masks_pil]

        # --- Convert back to Tensor ---
        image_tensor = TF.to_tensor(image_pil_resized) # C,H,W

        # Stack layer masks into a single tensor: (max_layers, H, W) torch.long
        target_masks_tensor = torch.stack(
            [torch.from_numpy(np.array(m_pil, dtype=np.int64)) for m_pil in layer_masks_pil_resized],
            dim=0
        )

        # --- Add Debugging Here ---
        if idx < 2: # Log for the first few samples
            logger.debug(f"Sample {idx} - Target Mask Stack Shape: {target_masks_tensor.shape}")
            background_id_check = SHAPE_TYPE_MAP.get("background", 0) # Get background ID
            for l_idx_check in range(self.max_layers_to_load):
                layer_mask_slice = target_masks_tensor[l_idx_check, :, :]
                num_foreground_pixels = torch.sum(layer_mask_slice != background_id_check).item()
                unique_vals = torch.unique(layer_mask_slice).cpu().numpy()
                logger.debug(f"  Layer {l_idx_check} - Sum: {layer_mask_slice.sum().item()}, Num FG Pixels: {num_foreground_pixels}, Unique: {unique_vals}")

        return image_tensor, target_masks_tensor

if __name__ == '__main__':
    # Example Usage:
    # Create a dummy dataset directory structure for testing
    # dummy_data_path = Path("./dummy_sem_data")
    # dummy_data_path.mkdir(exist_ok=True)
    # for i in range(3):
    #     sample_p = dummy_data_path / f"sem_{i:05d}"
    #     sample_p.mkdir(exist_ok=True)
    #     # Create dummy image and mask
    #     dummy_img = Image.new('L', (600, 400), color='gray')
    #     dummy_img.save(sample_p / "image_final_noisy_vis.png")
    #     dummy_mask = np.random.randint(0, 2, size=(400, 600), dtype=np.uint8)
    #     np.save(sample_p / "combined_actual_mask.npy", dummy_mask)

    print("Testing SEMDataset...")
    try:
        dataset = SEMDataset(data_dir='../generated_dataset', target_size=(256, 256), augment=True, use_tif=False) # Adjust path
        print(f"Dataset size: {len(dataset)}")
        img, msk = dataset[0]
        print(f"Sample 0 - Image shape: {img.shape}, Mask shape: {msk.shape}")
        print(f"Image dtype: {img.dtype}, Mask dtype: {msk.dtype}")
        print(f"Image min: {img.min()}, Image max: {img.max()}")
        print(f"Mask min: {msk.min()}, Mask max: {msk.max()}, Unique values: {torch.unique(msk)}")

        # Test with TIF if available
        # dataset_tif = SEMDataset(data_dir='../generated_dataset', target_size=(256, 256), use_tif=True)
        # print(f"TIF Dataset size: {len(dataset_tif)}")
        # img_tif, msk_tif = dataset_tif[0]
        # print(f"Sample 0 TIF - Image shape: {img_tif.shape}, Mask shape: {msk_tif.shape}")


    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure your generated_dataset path is correct and contains data.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Clean up dummy data if created
    # import shutil
    # if dummy_data_path.exists():
    #     shutil.rmtree(dummy_data_path)