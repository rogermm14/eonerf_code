# EO-NeRF
---
&#x231b; Warning &#x231b; This code has been in the fridge for a while. We are cleaning it up and documenting it little by little. Please be patient.

---
### [[Project page]](https://rogermm14.github.io/eonerf/)

&#128295; Developed at the [ENS Paris-Saclay, Centre Borelli](https://www.centreborelli.fr/) and accepted at the [CVPR EarthVision Workshop 2023](https://www.grss-ieee.org/events/earthvision-2023/).

This project follows our previous work [Sat-NeRF (2022)](https://centreborelli.github.io/satnerf/) and was recently used in &#128293; [S-EO (2025)](https://centreborelli.github.io/shadow-eo/) &#128293; to further leverage shadow predictions for improved 3D reconstructions from satellite images.

### [Multi-Date Earth Observation NeRF: The Detail Is in the Shadows](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Mari_Multi-Date_Earth_Observation_NeRF_The_Detail_Is_in_the_Shadows_CVPRW_2023_paper.pdf)
*[Roger Marí](https://scholar.google.com/citations?user=TgpSmIsAAAAJ&hl=en), 
[Gabriele Facciolo](http://dev.ipol.im/~facciolo/),
[Thibaud Ehret](https://tehret.github.io/)*

> **Abstract:** *We introduce Earth Observation NeRF (EO-NeRF), a new method for digital surface modeling and novel view synthesis from collections of multi-date remote sensing images. In contrast to previous variants of NeRF proposed in the literature for satellite images, EO-NeRF outperforms the altitude accuracy of advanced pipelines for 3D reconstruction from multiple satellite images, including classic and learned stereovision methods. This is largely due to a rendering of building shadows that is strictly consistent with the scene geometry and independent from other transient phenomena. In addition to that, a number of strategies are also proposed with the aim to exploit raw satellite images. We add model parameters to circumvent usual pre-processing steps, such as the relative radiometric normalization of the input images and the bundle adjustment for refining the camera models. We evaluate our method on different areas of interest using sets of 10-20 pre-processed and raw pansharpened WorldView-3 images.*

If you find this code or work helpful, please cite:
```
@inproceedings{mari2023multi,
  title={Multi-date earth observation nerf: The detail is in the shadows},
  author={Mar{\'\i}, Roger and Facciolo, Gabriele and Ehret, Thibaud},
  booktitle={2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  pages={2035--2045},
  year={2023}
}
```

---


## 1. Setup

To create the conda environment you can use the setup script, e.g.
```
conda init && bash -i setup_env.sh
```

Warning: If some libraries are not found, it may be necessary to update the environment variable `LD_LIBRARY_PATH` before launching the code:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
```
where `$CONDA_PREFIX` is the path to your conda or miniconda environment (e.g. `/mnt/cdisk/roger/miniconda3/envs/satnerf`).

---

## 2. Training

Example command to train EO-NeRF on the area of interest JAX_068 using the DFC2019 RGB crops:
```shell
(eonerf) $ bash run_JAX_RGB.sh JAX_068
```
Remember to update run_JAX_RGB.sh with your own data paths. Example data directories can be downloaded from [SatNeRF](https://github.com/centreborelli/satnerf/releases/tag/EarthVision2022).

---

## 3. Testing

Use the `eval_eonerf.py` script to generate the outputs of a pretrained EO-NeRF model. This script generates the learned dsm and rgb/shadow renderings. 

---


