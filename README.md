# TODO

- [x] simulation code
- [ ] train code
- [ ] evaluation code
- [ ] interactive simulation code

# EndoGSim: Physics-Aware 4D Dynamic Endoscopic Scene Simulations via MLLM-Guided Gaussian Splatting

### [Project Page](https://changjingliu.github.io/EndoGSim/) | [ArXiv](https://arxiv.org/abs/2605.16022)

#### [Changjing Liu](https://changjingliu.github.io/), [Yiming Huang](https://lastbasket.github.io/), [Long Bai](https://longbai-cuhk.github.io/), [Beilei Cui](https://beileicui.github.io/), [Hongliang Ren](https://www.ee.cuhk.edu.hk/en-gb/people/academic-staff/professors/prof-ren-hongliang)

<p align="left">
  <!-- <img width="60%" src="assets/teaser.png"/> -->
  <img width="80%" src="assets/teaser.gif"/>
</p>

## 1. Installation

```sh
conda create -n endogsim python=3.10 -y
conda activate endogsim

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
conda env update --file environment.yml

cd submodules
pip install ./simple-knn
pip install ./diff-plane-rasterization
```



## 2. Dataset



### Download .ply files

We provide three preprocessed datasets([Google Drive](https://drive.google.com/drive/folders/1ytB2xZtC2pAd9Sq3OfEfmUR6SHoiJ4-D?usp=sharing)). Please download them and place in the `./model` directory.
After downloading, the dataset structure will be as follows:

```
model/
├── cholecseg_sub/
│   └── video01_00080/
│       └── point_cloud/iteration_3000/
│                       └── point_cloud.ply
├── endonerf/
│   └── cutting_tissues_twice/
│       └── point_cloud/iteration_3000/
│                       └── point_cloud.ply
└── porcine_endo/
    └── gallbladder/
        └── point_cloud/iteration_3000/
                        └── point_cloud.ply
```

### Generate images

For example, to generate images for the `porcine_endo/gallbladder` dataset:

```sh
python simulation_gt.py \
  --model_path ./model/porcine_endo/gallbladder \
  --output_path ./model/porcine_endo/gallbladder \
  --physics_config ./config/porcine_endo/gallbladder_config_gt.json \
  --n_key_frame 10 \
  --dataset porcine_endo \
  --save_debug_flow  # optional
```

After generating, the dataset structure will be as follows:

```
model/
├── cholecseg_sub/
│   └── video01_00080/
│       ├── frames/
│       ├── images_generated/
│       └── point_cloud/iteration_3000/
|                        └── point_cloud.ply
├── endonerf/
│   └── cutting_tissues_twice/
│       ├── frames/
│       ├── images_generated/
│       └── point_cloud/iteration_3000/
|                        └── point_cloud.ply
└── porcine_endo/
    └── gallbladder/
        ├── frames/
        ├── images_generated/
        └── point_cloud/iteration_3000/
                        └── point_cloud.ply
```

## 3. Training

```sh
# cholecseg_sub dataset (from PAC-NeRF)
sh simulation_train_all_cholecseg_sub.sh

# endonerf dataset (from PhysDreamer)
sh simulation_train_all_endonerf.sh
```

---

## Simulation

Simulation can be performed by editing the configuration files to the corresponding config files.

```sh
python simulation_gt.py \
  --model_path ./model/porcine_endo/gallbladder \
  --output_path ./simulation_output/porcine_endo/gallbladder \
  --physics_config ./config/porcine_endo/my_config.json # modify this file to your own configuration\
  --n_key_frame 10 \
  --dataset porcine_endo \
  --save_debug_flow # optional
```

### Acknowledgements

This framework builds upon [PhysFlow](https://github.com/zhuomanliu/PhysFlow), [PhysGaussian](https://github.com/XPandora/PhysGaussian), [endo-4dgs](https://github.com/lastbasket/Endo-4DGS), [Pi^3](https://github.com/yyfz/Pi3).

---

