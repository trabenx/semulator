# sem_segmentation/train_infer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
import os
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src' / 'core')) # Add src/core to path


# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from constants import SHAPE_TYPE_MAP, NUM_SHAPE_CLASSES, MAX_PREDICTABLE_LAYERS
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


# --- Multi-class Dice for "X-Ray" output ---
def dice_coefficient_xray(model_outputs_flat, target_masks_stacked, epoch, num_shape_classes, max_layers, smooth=1e-5): # Increased smooth slightly
    # model_outputs_flat: (B, max_layers * num_shape_classes, H, W) - logits
    # target_masks_stacked: (B, max_layers, H, W) - with class indices (long)

    if num_shape_classes <= 1 or max_layers == 0:
        return torch.tensor(0.0, device=model_outputs_flat.device) # Ensure tensor is on correct device

    B, _, H, W = model_outputs_flat.shape
    
    # Reshape model output: (B, max_layers, num_shape_classes, H, W)
    model_outputs_reshaped = model_outputs_flat.view(B, max_layers, num_shape_classes, H, W)
    
    total_dice_score_for_batch = 0.0
    num_active_layers_in_batch = 0 # Count layers with foreground GT for averaging

    # --- Debug: Check input shapes and types once ---
    if not hasattr(dice_coefficient_xray, 'logged_shapes'):
        logger.debug(f"Dice - model_outputs_flat shape: {model_outputs_flat.shape}, dtype: {model_outputs_flat.dtype}")
        logger.debug(f"Dice - target_masks_stacked shape: {target_masks_stacked.shape}, dtype: {target_masks_stacked.dtype}")
        logger.debug(f"Dice - model_outputs_reshaped shape: {model_outputs_reshaped.shape}")
        dice_coefficient_xray.logged_shapes = True


    for l_idx in range(max_layers):
        layer_logits = model_outputs_reshaped[:, l_idx, :, :, :] # B, num_shape_classes, H, W
        layer_targets = target_masks_stacked[:, l_idx, :, :]   # B, H, W (long)

        # Only calculate Dice for layers that have actual foreground content in the target
        # This prevents penalizing for empty GT layers if model predicts something.
        # SHAPE_TYPE_MAP.get("background", 0) should be the ID for background class.
        background_class_id = SHAPE_TYPE_MAP.get("background", 0)

        # --- DEBUG: Check target content for this layer ---
        # This should be done for each item in batch, but for first item is a good start
        has_foreground_gt = torch.any(layer_targets[0] != background_class_id)
        logger.debug(f"Dice - BatchItem 0, Layer {l_idx}: Has FG in GT? {has_foreground_gt}. Unique GT labels: {torch.unique(layer_targets[0]).cpu().numpy()}")
        # ---
        
        if not torch.any(layer_targets != background_class_id):
            logger.debug(f"Dice - Layer {l_idx}: All background in target batch, skipping Dice for this layer.")
            continue
        num_active_layers_in_batch += 1

        # Get predicted class labels for this layer
        layer_pred_probs = torch.softmax(layer_logits, dim=1) # (B, num_shape_classes, H, W)
        layer_pred_labels = torch.argmax(layer_pred_probs, dim=1) # (B, H, W) with predicted class indices

        # --- Debug: Check labels ---
        if not hasattr(dice_coefficient_xray, f'logged_layer_{l_idx}'):
            logger.debug(f"Dice - Layer {l_idx} - Unique target labels: {torch.unique(layer_targets)}")
            logger.debug(f"Dice - Layer {l_idx} - Unique predicted labels: {torch.unique(layer_pred_labels)}")
            setattr(dice_coefficient_xray, f'logged_layer_{l_idx}', True)

        dice_per_class_this_layer_batch = [] # Store Dice for each class for this layer and batch

        for c in range(num_shape_classes):
            if c == background_class_id: # Typically skip background class for Dice
                continue

            # Create binary masks for the current class c
            pred_c = (layer_pred_labels == c).float() # (B, H, W)
            target_c = (layer_targets == c).float()   # (B, H, W)

            # Flatten for intersection/sum calculation
            pred_c_flat = pred_c.contiguous().view(B, -1)
            target_c_flat = target_c.contiguous().view(B, -1)

            intersection = (pred_c_flat * target_c_flat).sum(dim=1) # Sum over pixels for each item in batch
            sum_pred = pred_c_flat.sum(dim=1)
            sum_target = target_c_flat.sum(dim=1)
            
            # Dice score for class c, for each item in batch
            dice_score_class_batch = (2. * intersection + smooth) / (sum_pred + sum_target + smooth)
            
            # --- Debug: Check sums and intersection for a specific problematic class/layer ---
            if c == 1 and l_idx == 0 and epoch < 2: # Example: log for class 1, layer 0 in early epochs
                logger.debug(f"Epoch {epoch}, L{l_idx}, C{c} - Intersection: {intersection.cpu().numpy()}, SumPred: {sum_pred.cpu().numpy()}, SumTarget: {sum_target.cpu().numpy()}, Dice: {dice_score_class_batch.cpu().numpy()}")

            # Only consider Dice if the target actually has this class present (optional, but good for sparse classes)
            # If target_c.sum(dim=1) is 0 for an item, its Dice will be smooth / (pred.sum + smooth) which is low.
            # If both pred_c and target_c sum to 0, Dice is 1 (smooth/smooth). We want to avoid this.
            # We can average Dice scores for classes present in the target or average all.
            # For now, average all (non-background) per-class Dice scores for this layer.
            dice_per_class_this_layer_batch.append(dice_score_class_batch) # List of tensors, each (B,)

        if dice_per_class_this_layer_batch:
            # Stack and then mean over classes, then mean over batch
            avg_dice_this_layer_batch = torch.mean(torch.stack(dice_per_class_this_layer_batch, dim=0).mean(dim=1))
            total_dice_score_for_batch += avg_dice_this_layer_batch
        else:
            logger.debug(f"Dice - Layer {l_idx}: No foreground classes found in target, or all classes skipped.")


    # Average Dice score over active layers in the batch
    final_batch_dice = total_dice_score_for_batch / num_active_layers_in_batch if num_active_layers_in_batch > 0 else torch.tensor(0.0, device=model_outputs_flat.device)
    
    # --- Debug: Final Dice for batch ---
    logger.debug(f"Dice - Final Batch Dice: {final_batch_dice.item():.4f}, Active Layers: {num_active_layers_in_batch}")
    return final_batch_dice


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


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, checkpoint_dir, num_shape_classes, max_layers): # Added scheduler
    logger.info(f"Starting training for {epochs} epochs on {device} for {num_shape_classes} classes across {max_layers} layers...")
    best_val_dice = 0.0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch")
        for images, target_masks_stacked in progress_bar: # Renamed 'masks' to 'target_masks_stacked'
            images = images.to(device)
            target_masks_stacked = target_masks_stacked.to(device) # (B, max_layers, H, W)

            optimizer.zero_grad()
            outputs_flat = model(images) # (B, max_layers * num_shape_classes, H, W)

            # --- Calculate Loss and Dice PER LAYER ---
            B, _, H_out, W_out = outputs_flat.shape
            # Ensure NUM_SHAPE_CLASSES is not zero to avoid division error if constants not loaded
            current_num_shape_classes = num_shape_classes if num_shape_classes > 0 else 1
            outputs_reshaped = outputs_flat.view(B, max_layers, current_num_shape_classes, H_out, W_out)
            
            batch_total_loss = 0.0
            for l_idx in range(max_layers):
                layer_output_logits = outputs_reshaped[:, l_idx, :, :, :] # B, NUM_SHAPE_CLASSES, H, W
                layer_target_mask = target_masks_stacked[:, l_idx, :, :]    # B, H, W (long)
                batch_total_loss += criterion(layer_output_logits, layer_target_mask)
            
            # Average loss over layers for this batch
            current_batch_loss = batch_total_loss / max_layers if max_layers > 0 else batch_total_loss
            # --- End Per-Layer Loss ---

            # Dice coefficient is calculated on the full flat output and stacked targets
            # (dice_coefficient_xray should handle reshaping internally)
            current_batch_dice = dice_coefficient_xray(outputs_flat, target_masks_stacked, epoch, num_shape_classes, max_layers)

            current_batch_loss.backward() # Backpropagate the averaged loss
            optimizer.step()

            epoch_loss += current_batch_loss.item()
            epoch_dice += current_batch_dice.item() # .item() if dice is a tensor
            progress_bar.set_postfix(loss=current_batch_loss.item(), dice=current_batch_dice.item())

        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_dice = epoch_dice / len(train_loader)
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_epoch_loss:.4f}, Train Dice: {avg_epoch_dice:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", unit="batch")
            for images, target_masks_stacked in progress_bar_val: # Renamed
                images = images.to(device)
                target_masks_stacked = target_masks_stacked.to(device)
                outputs_flat = model(images)

                # --- Calculate Loss PER LAYER for Validation ---
                B_val, _, H_val, W_val = outputs_flat.shape
                outputs_reshaped_val = outputs_flat.view(B_val, max_layers, current_num_shape_classes, H_val, W_val)
                batch_total_loss_val = 0.0
                for l_idx in range(max_layers):
                    layer_output_logits_val = outputs_reshaped_val[:, l_idx, :, :, :]
                    layer_target_mask_val = target_masks_stacked[:, l_idx, :, :]
                    batch_total_loss_val += criterion(layer_output_logits_val, layer_target_mask_val)
                current_batch_loss_val = batch_total_loss_val / max_layers if max_layers > 0 else batch_total_loss_val
                # ---

                current_batch_dice_val = dice_coefficient_xray(outputs_flat, target_masks_stacked, epoch, num_shape_classes, max_layers)
                
                val_loss += current_batch_loss_val.item()
                val_dice += current_batch_dice_val.item() # .item() if dice is a tensor
                progress_bar_val.set_postfix(loss=current_batch_loss_val.item(), dice=current_batch_dice_val.item())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        logger.info(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")

        scheduler.step(avg_val_dice) # Step the scheduler based on validation Dice

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


def infer_single_image(model, image_path, device, target_size=(512, 512), use_tif=False, num_shape_classes=None, max_layers=5):
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
        output_logits_flat = model(image_tensor_resized) # (1, max_layers * num_shape_classes, H_resized, W_resized)

    B, _, H_r, W_r = output_logits_flat.shape
    output_logits_reshaped = output_logits_flat.view(B, max_layers, num_shape_classes, H_r, W_r)

    predicted_layer_masks_np = []
    for l_idx in range(max_layers):
        layer_logits = output_logits_reshaped[:, l_idx, :, :, :]
        # Get predicted class labels for this layer
        predicted_labels_resized_layer = torch.argmax(layer_logits, dim=1).squeeze(0).cpu() # H_resized, W_resized (long)

        # Resize mask back to original image dimensions
        pred_labels_resized_layer_for_tf = predicted_labels_resized_layer.unsqueeze(0).float() # Needs C dim, float
        layer_mask_original_size = TF.resize(
            pred_labels_resized_layer_for_tf,
            [original_h, original_w],
            interpolation=TF.InterpolationMode.NEAREST
        )
        predicted_layer_masks_np.append(layer_mask_original_size.squeeze(0).numpy().astype(np.uint8))
    
    # Returns a list of numpy arrays, each (OriginalH, OriginalW) with class IDs for that layer
    return predicted_layer_masks_np

def check_cuda_availability_details(chosen_device_str=None):
    """Prints details about CUDA availability and potential issues."""
    logger.info("--- Checking CUDA/Device Availability Details ---")
    try:
        if chosen_device_str == "cuda":
            if not torch.cuda.is_available():
                logger.error("User explicitly chose 'cuda', but torch.cuda.is_available() is False.")

                # 1. Check if PyTorch was compiled with CUDA support
                pytorch_cuda_version = torch.version.cuda
                if pytorch_cuda_version is None:
                    logger.warning("PyTorch was likely NOT compiled with CUDA support (torch.version.cuda is None).")
                else:
                    logger.info(f"PyTorch was compiled with CUDA version: {pytorch_cuda_version}")
                try:
                    device_count = torch.cuda.device_count()
                    logger.info(f"torch.cuda.device_count(): {device_count}")
                except Exception as e:
                    logger.warning(f"Error calling torch.cuda.device_count(): {e}")

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
        elif chosen_device_str == "cpu":
            logger.info("User explicitly chose 'cpu'.")
            return False # Indicates CUDA should not be used
        else: # Auto-detection
            if not torch.cuda.is_available():
                logger.warning("torch.cuda.is_available() returned False (auto-detection). CUDA is not available.")
                return False # CUDA not available
            else:
                logger.info("torch.cuda.is_available() returned True (auto-detection). CUDA will be used.")
                return True # CUDA available
    except Exception as e:
        logger.error(f"An unexpected error occurred during CUDA availability check: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Train/Infer for SEM X-Ray Segmentation.")
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
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help="Device to use: 'auto' (try CUDA, fallback to CPU), 'cuda' (force CUDA), 'cpu' (force CPU).")
    parser.add_argument('--max_layers', type=int, default=MAX_PREDICTABLE_LAYERS, help="Max layers model predicts for X-Ray vision.")

    args = parser.parse_args()

    # --- Setup Basic Logging (if not already configured globally) ---
    if not logger.handlers: # Configure only if no handlers are already set
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Determine device (CUDA or CPU) with detailed check and user choice ---
    chosen_device_cli = args.device.lower() # User's choice from CLI
    
    if chosen_device_cli == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("CUDA selected by user and available.")
            check_cuda_availability_details("cuda") # Log details
        else:
            logger.error("User selected 'cuda' but CUDA is not available! Please check setup. Falling back to CPU.")
            check_cuda_availability_details("cuda") # Log details about why it's not available
            device = torch.device("cpu")
    elif chosen_device_cli == "cpu":
        device = torch.device("cpu")
        logger.info("CPU selected by user.")
        check_cuda_availability_details("cpu") # Log (will confirm no CUDA attempt)
    else: # 'auto' mode
        cuda_is_ready_auto = check_cuda_availability_details() # This will print detailed info
        if cuda_is_ready_auto:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    logger.info(f"--- Using device: {device} ---")
    # ---

    target_size_tuple = (args.img_size, args.img_size)
    # --- Ensure constants are loaded (NUM_SHAPE_CLASSES) ---
    # This path adjustment assumes constants.py is in ../src/core relative to train_infer.py
    sys.path.append(str(Path(__file__).resolve().parent.parent / 'src' / 'core'))
    try:
        from constants import SHAPE_TYPE_MAP, NUM_SHAPE_CLASSES
        logger.info(f"Number of shape classes (including background): {NUM_SHAPE_CLASSES}")
    except ImportError:
        logger.error("Could not import SHAPE_TYPE_MAP, NUM_SHAPE_CLASSES from constants.")
        return
    # ---

    if args.mode == 'train':
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        dataset = SEMDataset(data_dir=args.data_dir, target_size=target_size_tuple,
                             augment=True, use_tif=args.use_tif,
                             num_classes=NUM_SHAPE_CLASSES, max_layers_to_load=args.max_layers) # Pass max_layers
        
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == 'cuda'))
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == 'cuda'))

        model = UNet(n_channels=1, n_total_output_channels=args.max_layers * NUM_SHAPE_CLASSES).to(device) # Correct output channels
        if args.model_path and Path(args.model_path).is_file():  # Check if resuming/fine-tuning
            try:
                logger.info(f"Floading model weights from: {args.model_path} for continued training.")
                model.load_state_dict(torch.load(args.model_path, map_location=device))
            except Exception as e:
                logger.error(f"Could no  load weights from {args.model_path}: {e}. Starting from scratch")
        criterion = nn.CrossEntropyLoss() # Ignores background by default if target has it and not in output channel for background
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, threshold=0.001, threshold_mode='abs', min_lr=1e-8) # Reduce LR if val_dice doesn't improve for 5 epochs
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args.epochs, Path(args.checkpoint_dir), NUM_SHAPE_CLASSES, args.max_layers)

    elif args.mode == 'infer':
        if not args.model_path or not args.input_image:
            logger.error("For inference, --model_path and --input_image are required.")
            return

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        model = UNet(n_channels=1, n_total_output_channels=args.max_layers * NUM_SHAPE_CLASSES).to(device)
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            logger.info(f"Loaded model from {args.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return

        list_of_predicted_layer_masks_np = infer_single_image(
            model, args.input_image, device, target_size_tuple, args.use_tif,
            num_shape_classes=NUM_SHAPE_CLASSES, max_layers=args.max_layers
        )

        if list_of_predicted_layer_masks_np:
            input_filename = Path(args.input_image).stem
            output_dir_path = Path(args.output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)

            #from src.core.utils import create_color_visualization, get_distinct_colors # For vis
            from utils import create_color_visualization, get_distinct_colors

            # --- Save individual predicted layer masks and their visualizations ---
            layer_vis_colors = get_distinct_colors(NUM_SHAPE_CLASSES) # Colors for shape types
            shape_type_colormap = {i: layer_vis_colors[i % len(layer_vis_colors)] for i in range(NUM_SHAPE_CLASSES)}

            for l_idx, pred_mask_np in enumerate(list_of_predicted_layer_masks_np):
                out_path_npy = output_dir_path / f"{input_filename}_pred_layer_{l_idx:02d}_semantic.npy"
                np.save(out_path_npy, pred_mask_np)
                logger.info(f"Saved predicted semantic mask for layer {l_idx} to {out_path_npy}")

                if np.any(pred_mask_np > 0): # Only visualize if there's content
                    pred_mask_vis_img = create_color_visualization(pred_mask_np, shape_type_colormap)
                    out_path_vis = output_dir_path / f"{input_filename}_pred_layer_{l_idx:02d}_semantic_vis.png"
                    iio.imwrite(out_path_vis, pred_mask_vis_img)

            # --- Load Original Image for Overlay ---
            original_image_for_overlay_np = None
            try:
                if args.use_tif:
                    original_image_np_load = iio.imread(args.input_image)
                    if original_image_np_load.ndim == 2: original_image_np_load = np.stack([original_image_np_load]*3, axis=-1)
                    elif original_image_np_load.shape[-1] == 1: original_image_np_load = np.repeat(original_image_np_load, 3, axis=-1)
                    elif original_image_np_load.shape[-1] == 4: original_image_np_load = original_image_np_load[:,:,:3]
                    if original_image_np_load.dtype == np.uint16: original_image_np_load = (original_image_np_load / 256).astype(np.uint8)
                    elif original_image_np_load.dtype != np.uint8:
                        min_val, max_val = np.min(original_image_np_load), np.max(original_image_np_load)
                        if max_val > min_val: original_image_np_load = ((original_image_np_load - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                        else: original_image_np_load = np.full_like(original_image_np_load, 128, dtype=np.uint8)
                else:
                    original_image_pil_load = Image.open(args.input_image).convert("RGB")
                    original_image_np_load = np.array(original_image_pil_load)
                
                # Ensure original image matches the (potentially resized) output mask dimensions
                # list_of_predicted_layer_masks_np[0] gives the shape of predicted masks (H,W)
                if original_image_np_load.shape[:2] != list_of_predicted_layer_masks_np[0].shape[:2]:
                    from skimage.transform import resize as sk_resize
                    logger.warning(f"Original image shape {original_image_np_load.shape[:2]} differs from predicted mask shape {list_of_predicted_layer_masks_np[0].shape[:2]}. Resizing original for overlay.")
                    original_image_for_overlay_np = sk_resize(original_image_np_load, list_of_predicted_layer_masks_np[0].shape[:2], preserve_range=True, anti_aliasing=True).astype(np.uint8)
                else:
                    original_image_for_overlay_np = original_image_np_load.copy()

            except Exception as e_load_orig:
                logger.warning(f"Could not load original image for overlay: {e_load_orig}")
                original_image_for_overlay_np = None


            # --- 1. Composite Overlay (Top-most with transparency) ---
            if original_image_for_overlay_np is not None:
                composite_overlay_image = original_image_for_overlay_np.copy()
                # Define a set of distinct colors for layers (not shape types within layer)
                layer_overlay_colors = get_distinct_colors(args.max_layers)
                alpha_blend = 0.5 # Transparency for overlay

                # Iterate from bottom layer to top layer for correct "drawing" order
                for l_idx in range(args.max_layers):
                    pred_mask_np_layer = list_of_predicted_layer_masks_np[l_idx]
                    layer_color = layer_overlay_colors[l_idx % len(layer_overlay_colors)]

                    # Find all foreground pixels for this layer (any shape type > background)
                    foreground_pixels_this_layer = (pred_mask_np_layer != SHAPE_TYPE_MAP.get("background", 0))
                    
                    if np.any(foreground_pixels_this_layer):
                        composite_overlay_image[foreground_pixels_this_layer] = (
                            (1 - alpha_blend) * composite_overlay_image[foreground_pixels_this_layer] +
                            alpha_blend * np.array(layer_color, dtype=np.uint8)
                        ).astype(np.uint8)
                
                output_composite_overlay_path = output_dir_path / f"{input_filename}_composite_overlay.png"
                iio.imwrite(output_composite_overlay_path, composite_overlay_image)
                logger.info(f"Saved composite overlay image to {output_composite_overlay_path}")


            # --- 2. Combined Layer-Colored Semantic Mask Visualization ---
            # Create a single mask where pixel color is determined by the topmost *active* layer's color
            # (using the same layer_overlay_colors)
            if list_of_predicted_layer_masks_np:
                h_pred, w_pred = list_of_predicted_layer_masks_np[0].shape
                combined_vis_mask_colored = np.zeros((h_pred, w_pred, 3), dtype=np.uint8) # Black background
                
                # Iterate from bottom layer to top layer, so top layer overwrites
                for l_idx in range(args.max_layers):
                    pred_mask_np_layer = list_of_predicted_layer_masks_np[l_idx]
                    layer_color = layer_overlay_colors[l_idx % len(layer_overlay_colors)]
                    
                    # Pixels that belong to any foreground shape type in this layer
                    foreground_pixels_this_layer = (pred_mask_np_layer != SHAPE_TYPE_MAP.get("background", 0))
                    
                    if np.any(foreground_pixels_this_layer):
                        combined_vis_mask_colored[foreground_pixels_this_layer] = layer_color
                
                output_combined_vis_path = output_dir_path / f"{input_filename}_combined_layers_vis.png"
                iio.imwrite(output_combined_vis_path, combined_vis_mask_colored)
                logger.info(f"Saved combined layer visualization to {output_combined_vis_path}")


if __name__ == '__main__':
    # Add a small delay to allow logger time to initialize in some environments
    time.sleep(0.1)
    main()
