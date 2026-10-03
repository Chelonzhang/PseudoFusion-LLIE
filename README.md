# Zero-Shot Low-Light Image Enhancement

This repository is the open-source implementation of our paper method. The method
operates in a **zero-shot** setting: it performs test-time optimization independently
for each input image, requiring **no paired training data**.

## Method Overview

```
Low-light input image
  └─ Preprocessing
      └─ SRIE Retinex decomposition (MATLAB) → reflectance R / illumination L
          └─ Pseudo-Multi-Frame Sequence Fusion Module
             (pseudo multi-frame generation + RAFT optical-flow alignment
              + hierarchical edge-preserving fusion) → R_fusion
              └─ Vision-Texture Feature Modulation Module
                 (Vision-LSTM dual-head refinement of the reflectance residual
                  and the illumination curve parameters) → enhanced image
```

The method is organized into two modules:

- **Pseudo-Multi-Frame Sequence Fusion Module** — generates a pseudo multi-frame
  sequence from the reflectance component, aligns the frames with RAFT optical flow,
  and fuses them into a single clean reflectance `R_fusion`
- **Vision-Texture Feature Modulation Module** — applies a Vision-LSTM dual-head
  refinement that predicts both a reflectance residual and the illumination curve
  parameters
## Requirements

- Python 3.8+
- PyTorch + torchvision
- numpy, opencv-python, einops
- **MATLAB** (R2018b or later, with the MATLAB Engine for Python installed)
  - The SRIE decomposition is implemented in MATLAB (`srie.m`, `processing.p`)
    and invoked through `matlab.engine`.

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Install the MATLAB Engine (run from the MATLAB installation directory; adjust the
path for your version):

```bash
cd "matlabroot/extern/engines/python"
python setup.py install
```

> On first run, `model_retinex.py` automatically starts the MATLAB engine and
> `addpath`s this directory so that `srie_wrapper.m` can be called.

## Usage (single image)

```bash
python demo.py --input path/to/low_light.jpg --output output
```

Alternatively, edit `INPUT_PATH` / `OUTPUT_DIR` at the top of `demo.py` and run:

```bash
python demo.py
```

Output: `output/<image_name>_enhanced.png` (the final enhanced result).

> Note: this method performs optimization per image,
> so processing a single image takes some time (depending on hardware and resolution).

## Citation

If you use this code, please cite the corresponding paper.
(Vision-LSTM is from NXAI; see the license header in `vision_lstm2.py`.)
