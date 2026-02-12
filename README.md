# ComfyUI Custom Node — Phase Congruency Edge

A ComfyUI custom node that detects edges and features using **Phase Congruency**, an illumination-invariant and contrast-invariant feature detector based on the coherence of Fourier phase components across multiple scales and orientations (Kovesi, 1999).

---

## What is Phase Congruency?

Most classical edge detectors (Sobel, Canny, Laplacian) rely on **intensity gradients**, so their output changes with lighting conditions and local contrast. Phase Congruency takes a fundamentally different approach:

> Features are detected at locations where Fourier components are **maximally in phase** with one another — regardless of their amplitude.

This means the detector is:

- **Illumination invariant** — absolute brightness has no effect
- **Contrast invariant** — works equally on high-contrast and low-contrast regions
- **Perceptually meaningful** — closely matches how the human visual system perceives edges and corners
- **Theoretically grounded** — based on the Kovesi (1999) frequency-domain formulation using log-Gabor filters

---

## Node

| Property | Value |
|---|---|
| Node name | `Phase Congruency Edge` |
| Category | `image/filters` |
| Input | `IMAGE` (RGB or grayscale, any resolution) |
| Output | `IMAGE` (grayscale edge map, white = strong edges) |

---

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `image` | IMAGE | — | — | Input image tensor |
| `nscale` | INT | 4 | 2–8 | Number of log-Gabor filter scales |
| `norient` | INT | 6 | 2–8 | Number of filter orientations |
| `min_wavelength` | INT | 3 | 2–20 | Minimum wavelength of the finest-scale filter (pixels) |
| `mult` | FLOAT | 2.1 | 1.5–4.0 | Multiplicative factor between successive filter scales |
| `sigma_on_f` | FLOAT | 0.55 | 0.1–1.0 | Bandwidth of each log-Gabor filter (ratio σ/f₀) |
| `k` | FLOAT | 2.0 | 0.5–10.0 | Noise threshold in standard deviations above the mean |
| `cutoff` | FLOAT | 0.5 | 0.1–0.9 | Butterworth low-pass filter cutoff frequency |

---

## Algorithm Overview

For each of the `norient` orientations:

1. Build a **log-Gabor** bandpass filter at each of the `nscale` scales, multiplied by an angular spread function (raised cosine) centered on the current orientation.
2. Apply each filter in the **frequency domain** (via FFT) to obtain complex-valued responses.
3. Sum the complex responses across all scales to form an **energy vector** `(E_real, E_imag)`.
4. Estimate the **noise threshold** `T` using Rayleigh statistics on the finest-scale amplitude:
   ```
   T = mean_noise + k × noise_std
   ```
5. Compute per-orientation phase congruency:
   ```
   PC_orient = max(|energy| − T, 0) / (Σ amplitude + ε)
   ```

Final phase congruency map = **mean of PC_orient across all orientations**, normalized to [0, 1].

---

## Parameter Tuning Guide

| Goal | Adjustment |
|------|-----------|
| Detect finer / more detailed edges | Decrease `min_wavelength` |
| Detect coarser / broader edges | Increase `min_wavelength` |
| More sensitive (at risk of noise) | Decrease `k` (e.g. 1.0) |
| Cleaner output, fewer false edges | Increase `k` (e.g. 4.0–6.0) |
| Improve isotropy (all directions) | Increase `norient` (e.g. 8) |
| Capture more scale information | Increase `nscale` (e.g. 6) |
| Reduce ringing / high-freq artefacts | Decrease `cutoff` (e.g. 0.4) |

**Recommended starting point:** `nscale=4`, `norient=6`, `min_wavelength=3`, `mult=2.1`, `sigma_on_f=0.55`, `k=2.0`

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/bemoregt/ComfyUI_CustomNode_PhaseCongruencyEdge.git
```

Then restart ComfyUI. The node will appear under **`image/filters`** → **Phase Congruency Edge**.

### Dependencies

Only standard packages are required — no extra `pip install` step needed if ComfyUI is already running:

```
numpy
pillow
torch
```

---

## Comparison with Other Edge Detectors

| Method | Illumination Invariant | Contrast Invariant | Multi-scale | Notes |
|--------|----------------------|-------------------|-------------|-------|
| Sobel | No | No | No | Fast, simple gradient |
| Canny | No | Partial | No | Needs manual threshold |
| Laplacian of Gaussian | No | No | No | Sensitive to noise |
| **Phase Congruency** | **Yes** | **Yes** | **Yes** | Perceptually meaningful |

---

## Example Use Cases

- Preprocessing for segmentation pipelines
- Artistic / stylized edge rendering
- Document scanning and OCR preprocessing
- Medical image feature extraction
- Texture analysis

---

## Reference

Kovesi, P. (1999). *Image Features from Phase Congruency*. **Videre: Journal of Computer Vision Research**, 1(3).

---

## License

MIT License — free for personal and commercial use.
