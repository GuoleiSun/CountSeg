# Object Counting and Instance Segmentation with Image-level Supervision, in CVPR 2019

[Paper](https://arxiv.org/abs/1903.02494) [Supp]() [Presentation]() [Poster]()

![block images](https://github.com/GuoleiSun/CountSeg/blob/master/demo/images/block.png)


## Requirements:
1. System: ubuntu 16.04. 
2. NVIDIA GPU + CUDA CuDNN
3. Python>=3.6
4. Pytorch version 0.4 
5. Jupyter Notebook and ipywidgets 
6. Other common packages: numpy, scipy, and so on. Please refer to environment.yml.

## Installation:
This respority uses some functions from [PRM](https://github.com/ZhouYanzhao/PRM).
1. Download this respority and unzip it. Make sure that the folders look like this:
```
  CountSeg
  ├── Nest-pytorch
      ├── ...
  ├── PRM-pytorch
      ├── ...
  ├── ...
```
2. Go inside to "CountSeg" folder by "cd path/CountSeg", where path is where you store CountSeg in your computer.
3. Install [Nest](https://github.com/ZhouYanzhao/Nest), a flexible tool for building and sharing deep learning modules, created by Yanzhao
   ```
   pip install git+https://github.com/ZhouYanzhao/Nest.git
   ```
4. Install PRM via Nest's CLI tool
   ```
   nest module install ./PRM-pytorch prm
   ```
   Validate the installation by "nest module list --filter prm", you should see something like this.
   ```
   ## Output:
   #
   # 3 Nest modules found.
   #[0] prm.fc_resnet50 (1.0.0)
   #[1] prm.peak_response_mapping (1.0.0)
   #[2] prm.prm_visualize (1.0.0)
   ```
   If you get some error, it is because that you miss some packages. Install them and do the validation again until you can get something like above
4. Install Nest's build-in Pytorch modules
   ```
   nest module install ./Nest-pytorch pytorch
   ```
   
## Dataset
1. For Pascal, download dataset by simply running script in CountSeg/dataset folder. 
   ```
   cd path/CountSeg/dataset
   chmod +x pascal_download.sh
   ./pascal_download.sh
   ```
   Before running the script, do not forget to change the save_path_2007 and save_path_2012 in pascal_download.sh to the path where you want to save pascal 2007 and 2012 dataset.
2. For COCO dataset, please download dataset from [COCO](http://cocodataset.org/#download)

## Demo 
Go to "CountSeg/demo" and run demo.

## Test
To reproduce the results reproted in the paper.
1. Pascal 2007 counting
   ```
   cd path/CountSeg
   jupyter notebook
   ```
   Then open eval_counting_pascal07.ipynb and simply run each cell inside it. Make sure you change the data path in eval_counting_pascal07.ipynb.
2. COCO 2014 counting. Open jupyter notebook and run eval_counting_coco14.ipynb. Make sure you change the data path in eval_counting_coco14.ipynb.

## Training
1. Pascal 2007 and COCO 2014.
   ```
   cd path/CountSeg/experiments
   jupyter notebook
   ```
   Then open main-pascal.ipynb or main-coco.ipynb to do training on Pascal or COCO, respectively. Make sure you change the data path in config_counting_pascal07.yml and config_counting_coco14.yml.
   
## Citation 
If you find the code useful for your research, please cite:
```bibtex
@INPROCEEDINGS{cholakkal_sun2019object,
    author = {Cholakkal, Hisham and Sun, Guolei and Khan, Fahad Shahbaz and Shao, Ling},
    title = {Object Counting and Instance Segmentation with Image-level Supervision},
    booktitle = {CVPR},
    year = {2019}
}
```
