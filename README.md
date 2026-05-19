# TODO
- [ ] code

# EndoGSim: Physics-Aware 4D Dynamic Endoscopic Scene Simulations via MLLM-Guided Gaussian Splatting

### <p align="left">[Project Page]() | [ArXiv](https://arxiv.org/abs/2605.16022)</p>
####  <p align="left"> Changjing Liu, Yiming Huang, Long Bai, Beilei Cui, Hongliang Ren*</p>



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
We provide three preprocessed datasets: Please download them and place in the `./model` directory.

After downloading, the dataset structure will be as follows:
```
model/
├── cholecseg_sub/
│   ├── video01_00080/
│   │   ├── frames
│   │   ├── images_generated
│   │   ├── point_cloud
├── endonerf/
│   ├── cutting_tissues_twice/
│   │   ├── images
│   │   ├── images_generated
│   │   ├── point_cloud
│   │   ...
```

## 3. Running
```sh
# for cholecseg_sub dataset from PAC-NeRF
sh simulation_train_all_cholecseg_sub.sh

# for endonerf dataset from PhysDreamer
sh simulation_train_all_endonerf.sh

```
---

### Acknowledgements

This framework builds upon​​  [PhysFlow](https://github.com/zhuomanliu/PhysFlow), [PhysGaussian](https://github.com/XPandora/PhysGaussian), [endo-4dgs](https://github.com/lastbasket/Endo-4DGS), [Pi^3](https://github.com/yyfz/Pi3).​​

---

