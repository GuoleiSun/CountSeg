# weaking_supervised_counting_ins_segmentation
Object counting and instance segmentation with image-level supervision, in CVPR 2019

![block images](https://github.com/GuoleiSun/CountSeg/blob/master/demo/images/block.png)


## Requirements:
1. System: ubuntu 16.04. 
2. NVIDIA GPU + CUDA CuDNN
3. Python>=3.6
4. Pytorch version 0.4 
5. Jupyter Notebook and ipywidgets 
6. Other common packages: numpy, scipy, and so on. Please refer to environment.yml.

## Installation:
Download this respority and unzip it. Make sure that the folders look like this:
   ```
  CountSeg
  ├── Nest-pytorch
      ├── ...
  ├── PRM-pytorch
      ├── ...
  ├── ...
  ```
1. Go inside to CountSeg folder by "cd path/CountSeg", where path is where you store CountSeg in your computer.
1. Install [Nest](https://github.com/ZhouYanzhao/Nest), a flexible tool for building and sharing deep learning modules, created by Yanzhao
   ```
   pip install git+https://github.com/ZhouYanzhao/Nest.git
   ```
2. Install PRM via Nest's CLI tool
   ```
   nest module install ./PRM-pytorch prm
   ```
   Validate the installation by "nest module list --filter prm", you should see something like this.
   ```
   #Output:
   #
   #3 Nest modules found.
   #[0] prm.fc_resnet50 (1.0.0)
   #[1] prm.peak_response_mapping (1.0.0)
   #[2] prm.prm_visualize (1.0.0)
   ```
   If you get some error, it is because that you miss some packages. Install them and do the validation again until you can ge t something like above
3. Install Nest's build-in Pytorch modules
   ```
   nest module install ./Nest-pytorch pytorch
   ```
