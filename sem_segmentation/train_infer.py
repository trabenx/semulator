# sem_segmentation/train_infer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import imageio.v3 as iio
import torchvision.transforms.functional as TF
from tqdm import tqdm
import logging
import time

from unet_model import UNet # Import your U-Net model
from dataset_loader import SEMDataset # Import your Dataset class

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src' / 'core')) # Add src/core to path


# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from constants import SHAPE_TYPE_MAP, NUM_SHAPE_CLASSES
except ImportError:
    logger.error("Could not import SHAPE_TYPE_MAP, NUM_SHAPE_CLASSES from constants. Ensure it's in src/core and path is correct.")
    # Define fallbacks if import fails, but this is not ideal
    NUM_SHAPE_CLASSES = 2 # Fallback: Binary (background + one foreground)
    SHAPE_TYPE_MAP = {"background": 0, "foreground": 1}



# --- Dice Loss (Common for Segmentation) ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred) # Apply sigmoid to get probabilities [0,1]
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice_score # We want to minimize 1 - DiceScore


# --- Multi-class Dice Coefficient (Example: Macro Average) ---
def dice_coefficient_multiclass(pred_logits, target_labels, num_classes, smooth=1e-6):
    if num_classes <= 1: # Fallback to binary if not really multiclass
        pred_probs = torch.sigmoid(pred_logits)
        pred_binary = (pred_probs > 0.5).float().view(-1)
        target_flat = target_labels.float().view(-1)
        intersection = (pred_binary * target_flat).sum()
        return (2. * intersection + smooth) / (pred_binary.sum() + target_flat.sum() + smooth)

    pred_probs = torch.softmax(pred_logits, dim=1) # Get probabilities per class
    pred_labels = torch.argmax(pred_probs, dim=1)  # Get predicted class label (B, H, W)

    dice_per_class = []
    for c in range(num_classes): # Iterate over each class
        if c == SHAPE_TYPE_MAP.get("background", 0): continue # Skip background for Dice usually

        pred_c = (pred_labels == c).float().view(-1)
        target_c = (target_labels == c).float().view(-1)

        intersection = (pred_c * target_c).sum()
        score = (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
        dice_per_class.append(score)

    if not dice_per_class: return torch.tensor(0.0, device=pred_logits.device) # Handle empty list
    return torch.mean(torch.stack(dice_per_class))


def dice_coefficient(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred_binary = (pred > 0.5).float() # Convert to binary predictions
    pred_flat = pred_binary.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, checkpoint_dir, num_classes):
    logger.info(f"Starting training for {epochs} epochs on {device} for {num_classes} classes...")
    best_val_dice = 0.0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch")
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            # dice = dice_coefficient(outputs, masks)
            dice = dice_coefficient_multiclass(outputs, masks, num_classes)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_dice += dice.item()
            progress_bar.set_postfix(loss=loss.item(), dice=dice.item())

        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_dice = epoch_dice / len(train_loader)
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_epoch_loss:.4f}, Train Dice: {avg_epoch_dice:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", unit="batch")
            for images, masks in progress_bar_val:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                # dice = dice_coefficient(outputs, masks)
                dice = dice_coefficient_multiclass(outputs, masks, num_classes)
                val_loss += loss.item()
                val_dice += dice.item()
                progress_bar_val.set_postfix(loss=loss.item(), dice=dice.item())


        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        logger.info(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_model_path = checkpoint_dir / f"best_model_epoch_{epoch+1}_dice_{avg_val_dice:.4f}.pth"
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved new best model to {best_model_path} (Val Dice: {best_val_dice:.4f})")
        
        # Save checkpoint periodically
        if (epoch + 1) % 5 == 0: # Save every 5 epochs
            chkpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), chkpt_path)
            logger.info(f"Saved checkpoint to {chkpt_path}")

    logger.info("Training finished.")


def infer_single_image(model, image_path, device, target_size=(512, 512), use_tif=False):
    model.eval()
    
    # Load and preprocess image (similar to dataset loader)
    try:
        if use_tif:
            image_np = iio.imread(image_path)
            if image_np.ndim == 2: image_np = image_np[:, :, np.newaxis]
            if image_np.dtype == np.uint16: image_np = image_np.astype(np.float32) / 65535.0
            elif image_np.dtype == np.uint8: image_np = image_np.astype(np.float32) / 255.0
            else: # Normalize other types if necessary
                 min_val, max_val = np.min(image_np), np.max(image_np)
                 if max_val > min_val: image_np = (image_np.astype(np.float32) - min_val) / (max_val - min_val)
                 else: image_np = np.zeros_like(image_np, dtype=np.float32)
        else: # PNG
            image_pil = Image.open(image_path).convert('L')
            image_np = np.array(image_pil, dtype=np.float32) / 255.0
            image_np = image_np[:, :, np.newaxis]
    except Exception as e:
        logger.error(f"Error loading image for inference {image_path}: {e}")
        return None

    image_tensor = TF.to_tensor(image_np)
    
    # Get original image dimensions for resizing output mask later if needed
    original_h, original_w = image_np.shape[0], image_np.shape[1]

    image_tensor_resized = TF.resize(image_tensor, list(target_size), interpolation=TF.InterpolationMode.BILINEAR)
    image_tensor_resized = image_tensor_resized.unsqueeze(0).to(device) # Add batch dimension

    with torch.no_grad():
        output_logits = model(image_tensor_resized)
        if num_classes == 1: # Binary case
            output_probs = torch.sigmoid(output_logits)
            predicted_labels_resized = (output_probs > 0.5).squeeze(0).cpu() # C, H, W -> H,W if C=1
        else: # Multi-class
            predicted_labels_resized = torch.argmax(output_logits, dim=1).squeeze(0).cpu() # B,H,W -> H,W (long)

    # Resize mask back to original image dimensions
    # Ensure predicted_labels_resized is (1,H,W) or (H,W) for resize
    if predicted_labels_resized.ndim == 2:
         predicted_labels_resized_for_tf = predicted_labels_resized.unsqueeze(0) # Add C dim for TF.resize
    else: # Should be (C,H,W) where C=1 for binary, or already H,W for argmax
         predicted_labels_resized_for_tf = predicted_labels_resized

    predicted_mask_original_size = TF.resize(
        predicted_labels_resized_for_tf.float(), # TF.resize needs float input
        [original_h, original_w],
        interpolation=TF.InterpolationMode.NEAREST
    )
    
    predicted_mask_np = predicted_mask_original_size.squeeze(0).numpy().astype(np.uint8)
    return predicted_mask_np


def check_cuda_availability_details():
    """Prints details about CUDA availability and potential issues."""
    logger.info("--- Checking CUDA Availability Details ---")
    try:
        if not torch.cuda.is_available():
            logger.warning("torch.cuda.is_available() returned False. CUDA is not available.")

            # 1. Check if PyTorch was compiled with CUDA support
            pytorch_cuda_version = torch.version.cuda
            if pytorch_cuda_version is None:
                logger.warning("PyTorch was likely NOT compiled with CUDA support (torch.version.cuda is None).")
                logger.warning("Ensure you installed a PyTorch version with CUDA (e.g., from pytorch.org for your CUDA version).")
            else:
                logger.info(f"PyTorch was compiled with CUDA version: {pytorch_cuda_version}")

            # 2. Check CUDA driver version (requires nvidia-smi, might not be available or on PATH)
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True, text=True, check=False)
                if result.returncode == 0 and result.stdout.strip():
                    driver_version = result.stdout.strip()
                    logger.info(f"NVIDIA driver version found: {driver_version}")
                else:
                    logger.warning("Could not run 'nvidia-smi' to check driver version. Is it installed and in PATH?")
                    if result.stderr:
                         logger.warning(f"nvidia-smi error: {result.stderr.strip()}")
            except FileNotFoundError:
                logger.warning("'nvidia-smi' command not found. Ensure NVIDIA drivers are installed and nvidia-smi is in PATH.")
            except Exception as e_smi:
                logger.warning(f"Error running nvidia-smi: {e_smi}")


            # 3. Check number of CUDA devices found by PyTorch
            # This might still be 0 even if PyTorch has CUDA, if drivers/toolkit are mismatched or no compatible GPU.
            try:
                device_count = torch.cuda.device_count()
                logger.info(f"torch.cuda.device_count() returned: {device_count}")
                if device_count == 0 and pytorch_cuda_version is not None:
                    logger.warning("PyTorch sees 0 CUDA devices, even though compiled with CUDA.")
                    logger.warning("Possible reasons: NVIDIA driver issue, CUDA toolkit mismatch with driver, or no compatible GPU found.")
                for i in range(device_count):
                    logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            except Exception as e_dev_count:
                logger.error(f"Error checking CUDA devices with PyTorch: {e_dev_count}")
                logger.warning("This might indicate a more severe problem with the CUDA setup or PyTorch installation.")


            # 4. Check CUDA_VISIBLE_DEVICES environment variable
            cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
            if cuda_visible_devices:
                logger.info(f"Environment variable CUDA_VISIBLE_DEVICES is set to: '{cuda_visible_devices}'.")
                if cuda_visible_devices == "-1":
                    logger.warning("CUDA_VISIBLE_DEVICES=-1 typically disables all GPUs for CUDA applications.")
            else:
                logger.info("Environment variable CUDA_VISIBLE_DEVICES is not set (or empty).")
            
            logger.info("Common reasons for CUDA not being available:")
            logger.info("  1. NVIDIA drivers not installed or not compatible with the CUDA toolkit.")
            logger.info("  2. CUDA Toolkit not installed or not compatible with the NVIDIA drivers.")
            logger.info("  3. PyTorch installed without CUDA support (e.g., CPU-only version).")
            logger.info("  4. CUDA_VISIBLE_DEVICES environment variable hiding the GPU(s).")
            logger.info("  5. Hardware issues with the GPU.")
            logger.info("Please verify your NVIDIA driver, CUDA Toolkit, and PyTorch installation.")

            return False
        else:
            logger.info("torch.cuda.is_available() returned True. CUDA should be available.")
            device_count = torch.cuda.device_count()
            logger.info(f"Number of CUDA devices found: {device_count}")
            for i in range(device_count):
                logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)} (CUDA Capability: {torch.cuda.get_device_capability(i)})")
            current_device_idx = torch.cuda.current_device()
            logger.info(f"Current CUDA device index: {current_device_idx}")
            return True
    except Exception as e:
        logger.error(f"An unexpected error occurred during CUDA availability check: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Train and run inference for SEM shape segmentation.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer'], help="Mode: 'train' or 'infer'")
    parser.add_argument('--data_dir', type=str, default='./generated_dataset', help="Path to the generated SEM dataset")
    parser.add_argument('--use_tif', action='store_true', help="Load .tif images instead of .png visuals for input")
    parser.add_argument('--img_size', type=int, default=256, help="Target size for resizing images (square HxW)")
    parser.add_argument('--epochs', type=int, default=25, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--val_split', type=float, default=0.15, help="Fraction of data for validation")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help="Directory to save model checkpoints")
    parser.add_argument('--model_path', type=str, help="Path to a trained model checkpoint for inference")
    parser.add_argument('--input_image', type=str, help="Path to a single image for inference")
    parser.add_argument('--output_dir', type=str, default='./results', help="Directory to save inference results")

    args = parser.parse_args()

    # --- Determine device (CUDA or CPU) with detailed check ---
    cuda_is_ready = check_cuda_availability_details()
    if cuda_is_ready:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"--- Using device: {device} ---") # This will now print after the detailed check
    # ---

    logger.info(f"Using device: {device}")
    logger.info(f"Number of shape classes (including background): {NUM_SHAPE_CLASSES}")

    target_size_tuple = (args.img_size, args.img_size)

    if args.mode == 'train':
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        dataset = SEMDataset(data_dir=args.data_dir, target_size=target_size_tuple,
                             augment=True, use_tif=args.use_tif, num_classes=NUM_SHAPE_CLASSES)
        
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = UNet(n_channels=1, n_classes=NUM_SHAPE_CLASSES).to(device)
        
        criterion = nn.CrossEntropyLoss() # For multi-class semantic segmentation
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        train_model(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, Path(args.checkpoint_dir), NUM_SHAPE_CLASSES)

    elif args.mode == 'infer':
        if not args.model_path or not args.input_image:
            logger.error("For inference, --model_path and --input_image are required.")
            return

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        model = UNet(n_channels=1, n_classes=NUM_SHAPE_CLASSES).to(device)
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            logger.info(f"Loaded model from {args.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return

        predicted_mask_np = infer_single_image(model, args.input_image, device, target_size_tuple, args.use_tif, num_classes=NUM_SHAPE_CLASSES)

        if predicted_mask_np is not None:
            input_filename = Path(args.input_image).stem
            output_mask_path_npy = Path(args.output_dir) / f"{input_filename}_predicted_semantic_mask.npy"
            np.save(output_mask_path_npy, predicted_mask_np) # Save raw class IDs
            logger.info(f"Saved predicted semantic mask (npy) to {output_mask_path_npy}")

            # --- Create Colored Visualization of Semantic Prediction ---
            try:
                from src.core.utils import create_color_visualization, get_distinct_colors
                # Use the SHAPE_TYPE_MAP for consistent coloring if possible, or just distinct colors
                num_colors_to_gen = max(np.max(predicted_mask_np) + 1, NUM_SHAPE_CLASSES)
                pred_vis_colors = get_distinct_colors(num_colors_to_gen)
                pred_colormap = {i: pred_vis_colors[i % len(pred_vis_colors)] for i in range(num_colors_to_gen)}
                predicted_mask_vis_img = create_color_visualization(predicted_mask_np, pred_colormap)

                output_mask_vis_path = Path(args.output_dir) / f"{input_filename}_predicted_semantic_mask_vis.png"
                iio.imwrite(output_mask_vis_path, predicted_mask_vis_img)
                logger.info(f"Saved predicted semantic mask visualization to {output_mask_vis_path}")
            except Exception as e_vis:
                 logger.warning(f"Could not create semantic mask visualization: {e_vis}")

            # --- Create Overlay on Original Image ---
            try:
                # Load original image (similar to infer_single_image loading)
                if args.use_tif:
                    original_image_np_load = iio.imread(args.input_image)
                    # ... (Convert TIF to 8-bit RGB as in infer_single_image) ...
                else:
                    original_image_pil_load = Image.open(args.input_image).convert("RGB")
                    original_image_np_load = np.array(original_image_pil_load)

                # Ensure original image is H, W, 3 and uint8
                if original_image_np_load.ndim == 2: original_image_np_load = np.stack([original_image_np_load]*3, axis=-1)
                if original_image_np_load.shape[-1] == 1: original_image_np_load = np.repeat(original_image_np_load, 3, axis=-1)
                if original_image_np_load.shape[-1] > 3: original_image_np_load = original_image_np_load[:,:,:3]
                if original_image_np_load.dtype != np.uint8:
                    if np.max(original_image_np_load) > 1.0: # Assume scaled 0-255 or 0-65535
                        original_image_np_load = (original_image_np_load / np.max(original_image_np_load) * 255).astype(np.uint8)
                    else: # Assume float 0-1
                         original_image_np_load = (original_image_np_load * 255).astype(np.uint8)


                # Predicted mask is H, W with class IDs
                # Create a colored overlay using the same colormap as the visualization
                overlay_image = original_image_np_load.copy()
                
                # Resize original image to match predicted mask's (original) dimensions if they differ
                # This assumes predicted_mask_np is already resized to original image size
                if original_image_np_load.shape[:2] != predicted_mask_np.shape[:2]:
                    from skimage.transform import resize as sk_resize
                    logger.warning(f"Original image shape {original_image_np_load.shape[:2]} differs from predicted mask shape {predicted_mask_np.shape[:2]}. Resizing original for overlay.")
                    original_image_np_load = sk_resize(original_image_np_load, predicted_mask_np.shape[:2], preserve_range=True, anti_aliasing=True).astype(np.uint8)
                    overlay_image = original_image_np_load.copy()


                # Alpha for blending
                alpha_blend = 0.4

                for class_id in range(1, NUM_SHAPE_CLASSES): # Iterate through foreground classes
                    class_mask_bool = (predicted_mask_np == class_id)
                    if np.any(class_mask_bool):
                        color_for_class = pred_colormap.get(class_id, (0,255,0)) # Default green
                        overlay_image[class_mask_bool] = (
                            (1 - alpha_blend) * overlay_image[class_mask_bool] +
                            alpha_blend * np.array(color_for_class, dtype=np.uint8)
                        ).astype(np.uint8)

                output_overlay_path = Path(args.output_dir) / f"{input_filename}_semantic_overlay.png"
                iio.imwrite(output_overlay_path, overlay_image)
                logger.info(f"Saved semantic overlay image to {output_overlay_path}")
            except Exception as e_overlay:
                logger.warning(f"Could not create semantic overlay image: {e_overlay}", exc_info=True)

if __name__ == '__main__':
    # Add a small delay to allow logger time to initialize in some environments
    time.sleep(0.1)
    main()