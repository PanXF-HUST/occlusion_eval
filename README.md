# occlusion_eval
This is a project for a bachelor degree. Occlusion evaluation based on keypoints detection and modeling based on part relationship

The baseline of the project is [AlphePose-pytorch](https://github.com/MVIG-SJTU/AlphaPose/tree/pytorch)

## Environment

- Ubuntu 16.04/18.04.
- python 3.5+.
- Pytorch 1.0.0+.
- other dependencies.


## Installation

1. Get the code.
  ```Shell
  git clone https://github.com/PanXF-HUST/occlusion_eval/edit/master
  ```

2. Install [pytorch 1.0.0](https://github.com/pytorch/pytorch)and the corresponding version of torchvision and other dependencies.
  ```Shell
  pip install -r requirements.txt
  ```

3. Download the models manually: **duc_se.pth** (2018/08/30) ([Google Drive]( https://drive.google.com/open?id=1OPORTWB2cwd5YTVBX-NE8fsauZJWsrtW) | [Baidu pan](https://pan.baidu.com/s/15jbRNKuslzm5wRSgUVytrA)), **yolov3-spp.weights**([Google Drive](https://drive.google.com/open?id=1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC) | [Baidu pan](https://pan.baidu.com/s/1Zb2REEIk8tcahDa8KacPNA)). Place them into `./models/sppe` and `./models/yolo` respectively.



## Quick Start
- **Input dir**:  Run Project for all images in a folder with:
```
python3 demo.py --indir ${img_directory} --outdir examples/res --save_img
```

## NOTE
This project is a multi-task integration, part occlusion evaluation is only one part of the project, you can change the [fn.py] to change the format of output
