# EO-NeRF Data

Description of the data used in "Multi-date Earth Observation NeRF: The Detail Is in the Shadows", presented at the CVPR 2023 EarthVision Workshop.

Any scientific publication using the data shall cite the EO-NeRF paper and the rest of publications mentioned in this README. 

```
@inproceedings{mari2023multi,
  title={Multi-date earth observation nerf: The detail is in the shadows},
  author={Mar{\'\i}, Roger and Facciolo, Gabriele and Ehret, Thibaud},
  booktitle={2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  pages={2035--2045},
  year={2023}
}
```

The available data comprises 3 sets: **(1) JAX_RGB**, **(2) JAX_NEW** and **(3) IARPA**. 7 different areas of interest are covered.

JAX_RGB was originally used in Sat-NeRF (Marí et al., 2022). EO-NeRF incorporated JAX_NEW and IARPA.


### (1) JAX_RGB set

- **Image dir:** `DFC2019/Track3-RGB-crops` -> Contains uint8 RGB images cropped from the DFC2019 preprocessed data.
- **JSON dir:** `SatNeRF/root_dir/crops_rpcs_ba_v2` -> Contains the satellite images metadata (geographic location, RPC models, sun position).
- **GT dir:** `Datasets/DFC2019/Track3-Truth` -> Contains ground-truth surface models (DSM) and segmentation masks (CLS) from the DFC2019 data.

### (2) JAX_NEW set

- **Image dir:** `DFC2019/Track3-NEW-crops` -> Contains float32 RGB images cropped and pansharpened from the DFC2019 raw data.
- **JSON dir:** `SatNeRF/root_dir/crops_rpcs_ba_v2` -> Contains the satellite images metadata (geographic location, RPC models, sun position).
- **GT dir:** `Datasets/DFC2019/Track3-Truth` -> Contains digital surface models (DSM) and segmentation masks (CLS) from the DFC2019 data.

### (3) IARPA set

- **Image dir:** `SatNeRF_IARPA/crops` -> Contains float32 RGB images cropped and pansharpened from the IARPA 2016 raw data.
- **JSON dir:** `SatNeRF_IARPA/root_dir/rpcs_ba` -> Contains the satellite images metadata (geographic location, RPC models, sun position).
- **GT dir:** `SatNeRF_IARPA/Truth` -> Contains digital surface models (DSM) and segmentation masks (CLS) from the IARPA 2016 data.


### Additional comments:

Only the image and json directories are needed to train EO-NeRF. GT data is only used for evaluation.

2025 extension: Shadow segmentation masks generated with 
"S-EO: A Large-Scale Dataset for Geometry-Aware Shadow Detection in Remote Sensing Applications" (Masquil et al., 2025) are included for shadow supervision, which can help improve the output 3D geometry. The shadow segmentation masks can be found in `DFC2019/Shadows-pred_v2`and `SatNeRF_IARPA/Shadows-pred_v2`.


## Original data sources:

   - **IARPA areas of interest:** IARPA_001, IARPA_002, IARPA_003

     IARPA 2016 MVS 3D Mapping Challenge:
     https://www.jhuapl.edu/satellite-benchmark.html

     Any scientific publication using the data shall refer to the following paper:
     ```
      @inproceedings{bosch2016multiple,
        title={A multiple view stereo benchmark for satellite imagery},
        author={Bosch, Marc and Kurtz, Zachary and Hagstrom, Shea and Brown, Myron},
        booktitle={2016 IEEE Applied Imagery Pattern Recognition Workshop (AIPR)},
        pages={1--9},
        year={2016},
        organization={IEEE}
      }
      ```

   - **JAX areas of interest:** JAX_004, JAX_068, JAX_214, JAX_260
     
     Data Fusion Contest 2019 (DFC2019):
     https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019

     Any scientific publication using the data shall refer to the following papers:
     ```
      @inproceedings{bosch2019semantic,
        title={Semantic stereo for incidental satellite images},
        author={Bosch, Marc and Foster, Kevin and Christie, Gordon and Wang, Sean and Hager, Gregory D and Brown, Myron},
        booktitle={2019 IEEE Winter Conference on Applications of Computer Vision (WACV)},
        pages={1524--1532},
        year={2019},
        organization={IEEE}
      }
      ```

     ```
      @article{le20192019,
        title={2019 Data Fusion Contest [Technical Committees]},
        author={Le Saux, Bertrand and Yokoya, Naoto and Hansch, Ronny and Brown, Myron and Hager, Greg},
        journal={IEEE Geoscience and Remote Sensing Magazine},
        volume={7},
        number={1},
        pages={103--105},
        year={2019},
        publisher={IEEE}
      }
      ```

The authors would like to thank the Johns Hopkins University Applied Physics Laboratory and IARPA for providing the data used in this study, and the IEEE GRSS Image Analysis and Data Fusion Technical Committee for organizing the Data Fusion Contest.

