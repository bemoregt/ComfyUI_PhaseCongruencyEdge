import torch
import numpy as np
from PIL import Image, ImageOps
import math


class PhaseCongruencyEdge:
    """
    ComfyUI Custom Node: Phase Congruency Edge Detection

    Detects edges using phase congruency — an illumination/contrast invariant
    feature detector based on the coherence of Fourier phase components
    across multiple scales and orientations (Kovesi, 1999).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "nscale": ("INT", {
                    "default": 4,
                    "min": 2,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Number of wavelet scales"
                }),
                "norient": ("INT", {
                    "default": 6,
                    "min": 2,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Number of filter orientations"
                }),
                "min_wavelength": ("INT", {
                    "default": 3,
                    "min": 2,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Minimum wavelength of log-Gabor filter"
                }),
                "mult": ("FLOAT", {
                    "default": 2.1,
                    "min": 1.5,
                    "max": 4.0,
                    "step": 0.1,
                    "tooltip": "Scaling factor between successive filter scales"
                }),
                "sigma_on_f": ("FLOAT", {
                    "default": 0.55,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Bandwidth of log-Gabor filter"
                }),
                "k": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "Noise threshold (std devs above mean)"
                }),
                "cutoff": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Fractional measure of frequency spread below which phase congruency values get penalized"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("phase_congruency_edge",)
    FUNCTION = "compute_phase_congruency"
    CATEGORY = "image/filters"

    def compute_phase_congruency(self, image, nscale, norient, min_wavelength,
                                  mult, sigma_on_f, k, cutoff):
        # image: [B, H, W, C] torch tensor, values in [0, 1]
        batch_results = []

        for b in range(image.shape[0]):
            img_np = image[b].cpu().numpy()  # [H, W, C]

            # Convert to grayscale float in [0, 1]
            if img_np.shape[2] == 1:
                gray = img_np[:, :, 0].astype(np.float64)
            else:
                # Luminance conversion
                gray = (0.299 * img_np[:, :, 0] +
                        0.587 * img_np[:, :, 1] +
                        0.114 * img_np[:, :, 2]).astype(np.float64)

            pc = self._phase_congruency(
                gray, nscale, norient, min_wavelength,
                mult, sigma_on_f, k, cutoff
            )

            # Normalize to [0, 1]
            pc_min, pc_max = pc.min(), pc.max()
            if pc_max - pc_min > 1e-10:
                pc = (pc - pc_min) / (pc_max - pc_min)
            else:
                pc = np.zeros_like(pc)

            # Convert to [H, W, 3] RGB tensor
            pc_rgb = np.stack([pc, pc, pc], axis=-1).astype(np.float32)
            batch_results.append(torch.from_numpy(pc_rgb))

        output = torch.stack(batch_results, dim=0)  # [B, H, W, 3]
        return (output,)

    def _phase_congruency(self, img, nscale, norient, min_wavelength,
                           mult, sigma_on_f, k, cutoff):
        """
        Computes phase congruency for a 2D grayscale image [0,1].
        Based on Kovesi's algorithm using log-Gabor filters.

        For each orientation:
          1. Apply log-Gabor filters at each scale
          2. Accumulate complex responses (energy vector) and amplitudes
          3. Estimate noise threshold using Rayleigh statistics
          4. PC_orient = max(|energy| - T, 0) / (sum_amplitude + epsilon)
        Final PC = mean over all orientations.
        """
        rows, cols = img.shape
        epsilon = 1e-4  # avoid division by zero

        # Frequency coordinate grids (centered, range -0.5..0.5)
        cx = (np.arange(cols) - cols // 2) / cols
        cy = (np.arange(rows) - rows // 2) / rows
        x, y = np.meshgrid(cx, cy)

        radius = np.sqrt(x**2 + y**2)
        radius[rows // 2, cols // 2] = 1.0  # avoid log(0) at DC

        theta = np.arctan2(-y, x)  # orientation of each frequency component

        # Butterworth low-pass filter to suppress high-frequency noise
        lp = self._lowpass_filter(rows, cols, 0.45, 15)

        # FFT of input image (shifted so DC is at center)
        img_fft = np.fft.fftshift(np.fft.fft2(img))

        # Accumulate per-orientation PC maps, then average
        pc_sum = np.zeros((rows, cols))

        for orient in range(norient):
            angle = orient * np.pi / norient  # filter orientation

            # Angular spread function (raised cosine)
            ds = np.sin(theta) * np.cos(angle) - np.cos(theta) * np.sin(angle)
            dc = np.cos(theta) * np.cos(angle) + np.sin(theta) * np.sin(angle)
            dtheta = np.abs(np.arctan2(ds, dc))
            dtheta = np.minimum(dtheta * norient / 2.0, np.pi)
            spread = (np.cos(dtheta) + 1.0) / 2.0

            sum_an = np.zeros((rows, cols))
            energy_real = np.zeros((rows, cols))
            energy_imag = np.zeros((rows, cols))
            an_finest = None  # amplitude at the finest (smallest) scale

            for scale in range(nscale):
                wavelength = min_wavelength * (mult ** scale)
                fo = 1.0 / wavelength

                # Log-Gabor radial component
                log_gabor = np.exp(
                    -(np.log(radius / fo)) ** 2 /
                    (2.0 * (np.log(sigma_on_f) ** 2))
                )
                log_gabor[rows // 2, cols // 2] = 0.0  # remove DC

                # Combined filter = radial × angular × low-pass
                filt = log_gabor * spread * lp

                # Complex filter response
                resp = np.fft.ifft2(np.fft.ifftshift(img_fft * filt))
                re, im = resp.real, resp.imag
                an = np.sqrt(re**2 + im**2)

                sum_an += an
                energy_real += re
                energy_imag += im

                if scale == 0:
                    an_finest = an  # save finest scale amplitude for noise est.

            # Rayleigh noise threshold from finest-scale amplitude statistics
            # E[amplitude] for Rayleigh: mean = tau * sqrt(pi/2)
            tau = np.sqrt(np.mean(an_finest**2) / 2.0 + epsilon)
            mean_noise = tau * math.sqrt(math.pi / 2.0)
            noise_std = tau * math.sqrt((4.0 - math.pi) / 2.0)
            T = mean_noise + k * noise_std

            # Phase congruency for this orientation
            energy = np.sqrt(energy_real**2 + energy_imag**2)
            pc_orient = np.maximum(energy - T, 0.0) / (sum_an + epsilon)
            pc_sum += pc_orient

        # Average over all orientations
        pc = pc_sum / norient
        return pc

    def _lowpass_filter(self, rows, cols, cutoff, n):
        """
        Butterworth low-pass filter in the frequency domain.
        cutoff: cutoff frequency as fraction of Nyquist (0..0.5)
        n: filter order
        """
        if cols % 2 == 0:
            cx = np.arange(-cols // 2, cols // 2) / cols
        else:
            cx = np.arange(-(cols - 1) // 2, (cols - 1) // 2 + 1) / cols

        if rows % 2 == 0:
            cy = np.arange(-rows // 2, rows // 2) / rows
        else:
            cy = np.arange(-(rows - 1) // 2, (rows - 1) // 2 + 1) / rows

        x, y = np.meshgrid(cx, cy)
        radius = np.sqrt(x**2 + y**2)
        lp = 1.0 / (1.0 + (radius / cutoff) ** (2 * n))
        return lp


NODE_CLASS_MAPPINGS = {
    "PhaseCongruencyEdge": PhaseCongruencyEdge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhaseCongruencyEdge": "Phase Congruency Edge"
}
