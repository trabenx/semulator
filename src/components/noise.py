import numpy as np
import random
from ..core.utils import get_rng
# from skimage.util import random_noise # Can use this too

# Need HAS_PERLIN check here too
try:
    from perlin_noise import PerlinNoise
    HAS_PERLIN = True
except ImportError:
    HAS_PERLIN = False

def add_gaussian_noise(image, params, rng):
    """Adds Gaussian noise."""
    sigma = params.get('sigma', 0.02) # Noise level relative to max intensity (1.0)
    # Use numpy RandomState derived from the generator's RNG for numpy calls
    np_rng = np.random.RandomState(rng.randint(0, 2**32 - 1))
    noise = np_rng.normal(0, sigma, image.shape).astype(np.float32)
    noisy_image = image + noise
    print(f"Added Gaussian noise: sigma={sigma:.3f}")
    return noisy_image, noise

def add_poisson_noise(image, params, rng):
    # Approximation: Gaussian noise with std dev = sqrt(intensity * scale)
    scale = params.get('scale', 0.1) # Adjust how much intensity affects noise variance
    variance = np.maximum(0, image) * scale # Variance proportional to signal, ensure non-negative
    sigma = np.sqrt(variance)
    np_rng = np.random.RandomState(rng.randint(0, 2**32 - 1))
    noise = np_rng.normal(0, 1, image.shape).astype(np.float32) * sigma # Apply spatially varying sigma
    noisy_image = image + noise
    print(f"Added Poisson-like noise (scale={scale:.2f})")
    return noisy_image, noise

def add_salt_pepper_noise(image, params, rng):
    """Adds Salt & Pepper noise."""
    prob = params.get('probability', 0.005)
    s_vs_p = 0.5
    noisy_image = image.copy()
    noise_map = np.zeros_like(image, dtype=np.float32)
    np_rng = np.random.RandomState(rng.randint(0, 2**32 - 1)) # Use numpy's RNG

    # Salt
    num_salt = np.ceil(prob * image.size * s_vs_p)
    coords = tuple([np_rng.randint(0, i - 1, int(num_salt)) for i in image.shape])
    salt_val = 1.0
    delta_salt = salt_val - noisy_image[coords]
    noisy_image[coords] = salt_val
    noise_map[coords] += delta_salt

    # Pepper
    num_pepper = np.ceil(prob * image.size * (1. - s_vs_p))
    coords = tuple([np_rng.randint(0, i - 1, int(num_pepper)) for i in image.shape])
    pepper_val = 0.0
    delta_pepper = pepper_val - noisy_image[coords]
    noisy_image[coords] = pepper_val
    noise_map[coords] += delta_pepper

    print(f"Added Salt & Pepper noise: prob={prob:.4f}")
    return noisy_image, noise_map


def add_sem_texture_noise(image, params, rng):
     """Adds procedural texture noise (e.g., Perlin), potentially anisotropic."""
     # Requires Perlin noise implementation (from library or manual)
     if not HAS_PERLIN: # Check if Perlin is available (from background.py import)
         print("Warning: Perlin noise library not found. Skipping SEM texture noise.")
         return image, np.zeros_like(image)

     scale = params.get('scale', 30.0)
     octaves = params.get('octaves', 5)
     contrast = params.get('contrast', 0.05)
     anisotropy = params.get('anisotropy', 1.0) # Ratio of scale Y / scale X

     noise_gen = PerlinNoise(octaves=octaves, seed=rng.randint(0, 10000))
     h, w = image.shape
     scale_x = scale
     scale_y = scale * anisotropy # Stretch noise vertically if anisotropy > 1

     texture = np.zeros((h, w), dtype=np.float32)
     for i in range(h):
         for j in range(w):
             texture[i][j] = noise_gen([i / scale_y, j / scale_x])

     texture = (texture - np.mean(texture))
     std_dev = np.std(texture)
     if std_dev > 1e-6:
         texture = texture / std_dev
     texture_noise = texture * contrast

     noisy_image = image + texture_noise
     print(f"Added SEM texture noise: scale={scale:.1f}, contrast={contrast:.3f}, anisotropy={anisotropy:.2f}")
     return noisy_image, texture_noise

def apply_quantization(image, params, rng):
    """Simulates reduction to a lower bit depth."""
    target_bits = params.get('target_bits', 8)
    num_levels = 2**target_bits

    # Quantize: scale to [0, levels-1], round, scale back to [0, 1]
    quantized_image = np.round(np.clip(image, 0.0, 1.0) * (num_levels - 1)) / (num_levels - 1)
    added_noise = quantized_image - image # The quantization error

    print(f"Applied quantization to {target_bits} bits.")
    return quantized_image, added_noise


def apply_blur_noise(image, params, rng):
     """Applies simple Gaussian blur as a form of noise/signal degradation."""
     sigma = params.get('sigma', 0.7)
     ksize = int(sigma * 6 + 1)
     if ksize % 2 == 0: ksize += 1

     blurred_image = cv2.GaussianBlur(image, (ksize, ksize), sigma)
     # Noise map here represents the difference (signal removed by blur)
     added_noise = blurred_image - image

     print(f"Applied blur noise: sigma={sigma:.2f}")
     return blurred_image, added_noise


# Fixed Pattern Noise is often applied additively like other noises,
# but can also be multiplicative. Added an implementation in instrument_optical.py
# If needed here as a separate step:
# def add_fixed_pattern_noise(...) -> handled in instrument_optical


# --- Add FPN ---

# --- Factory ---
def apply_noise(image, noise_type, params, rng):
    """Applies a selected noise type and returns noisy image + added noise map."""
    if noise_type == 'gaussian':
        return add_gaussian_noise(image, params, rng)
    elif noise_type == 'poisson':
        return add_poisson_noise(image, params, rng)
    elif noise_type == 'salt_pepper':
        return add_salt_pepper_noise(image, params, rng)
    elif noise_type == 'sem_texture':
        return add_sem_texture_noise(image, params, rng)
    elif noise_type == 'quantization':
         # Quantization is usually applied last, or just before final bit depth conversion
         return apply_quantization(image, params, rng)
    elif noise_type == 'blur_noise':
         return apply_blur_noise(image, params, rng)
    # elif noise_type == 'fixed_pattern': -> Handled in instrument stage
    #     return add_fixed_pattern_noise(image, params, rng)
    else:
        print(f"Warning: Noise type '{noise_type}' not implemented.")
        return image, np.zeros_like(image) # Return unchanged image and zero noise map
