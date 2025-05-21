# sem_segmentation/dataset_loader.py
import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter 
import imageio.v3 as iio # Use v3 for consistent API
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch
import random
from pathlib import Path

class SEMDataset(Dataset):
    def __init__(self, data_dir, target_size=(512, 512), augment=False, use_tif=False, num_classes=None):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.augment = augment
        self.use_tif = use_tif # If true, loads .tif, otherwise .png visual
        self.num_classes = num_classes

        self.sample_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('sem_')])
        if not self.sample_dirs:
            raise FileNotFoundError(f"No sample directories (sem_XXXXX) found in {data_dir}")

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
        # Mask: from H,W (int64) to PIL 'L' (labels will be preserved as pixel values if <256)
        # If num_classes > 255, this PIL conversion for mask augmentation is problematic for 'L' mode.
        # For now, assume num_classes < 256 for PIL-based mask augmentations.
        mask_pil = Image.fromarray(mask.astype(np.uint8), mode='L') # Mode 'L' for 8-bit labels

        # --- Augmentation (applied to PIL Images) ---
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image_pil = TF.hflip(image_pil)
                mask_pil = TF.hflip(mask_pil)
            # Random vertical flip
            if random.random() > 0.5:
                image_pil = TF.vflip(image_pil)
                mask_pil = TF.vflip(mask_pil)
            # Random rotation (0, 90, 180, 270 degrees)
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                image_pil = TF.rotate(image_pil, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask_pil = TF.rotate(mask_pil, angle, interpolation=TF.InterpolationMode.NEAREST)

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

            # Small Affine Transformation (Careful with masks - NEAREST needed)
            if random.random() > 0.2:
                affine_angle = random.uniform(-10, 10) # degrees
                max_translate = 0.1 * self.target_size[0] # 10% of width/height
                translate_x = random.uniform(-max_translate, max_translate)
                translate_y = random.uniform(-max_translate, max_translate)
                scale = random.uniform(0.9, 1.1)
                shear = random.uniform(-5, 5) # degrees

                image_pil = TF.affine(image_pil, angle=affine_angle, translate=(translate_x, translate_y),
                                      scale=scale, shear=shear, interpolation=TF.InterpolationMode.BILINEAR, fill=0) # Fill with black
                mask_pil = TF.affine(mask_pil, angle=affine_angle, translate=(translate_x, translate_y),
                                     scale=scale, shear=shear, interpolation=TF.InterpolationMode.NEAREST, fill=0) # Fill with background ID

        # --- Resize (after augmentations) ---
        image_pil_resized = TF.resize(image_pil, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask_pil_resized = TF.resize(mask_pil, self.target_size, interpolation=TF.InterpolationMode.NEAREST)

        # --- Convert back to Tensor ---
        # Image: PIL 'L' to Tensor (C,H,W) float [0,1]
        image_tensor = TF.to_tensor(image_pil_resized)
        # Mask: PIL 'L' (with class labels) to Tensor (H,W) long
        mask_tensor = torch.from_numpy(np.array(mask_pil_resized, dtype=np.int64))

        return image_tensor, mask_tensor

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