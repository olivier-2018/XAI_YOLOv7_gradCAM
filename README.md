# GradCAM on Yolov7
Original Yolo_v7 release by WongKinYiu. 
Ref: https://github.com/WongKinYiu/yolov7.git

## Getting started:
- Install poetry (manages dependances for pip):
```sh
curl -sSL https://install.python-poetry.org | python3 -
```
- Configure poetry to install virtual environment inside project folders:
```sh
poetry config virtualenvs.in-project true
```
- Clone the repository:
```sh
git clone git@github.com:olivier-2018/XAI_YOLOv7_gradCAM.git
# then
cd XAI_YOLOv7_gradCAM 
```
- Download yolov7 pre-trained weights 
```sh
mkdir weights
cd weights 
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```
- Create a poetry virtual environment within the repository:
```sh
poetry install
```
- Activate the virtual environment:
```sh
poetry shell
```

### Inference with Yolov7 (see official site in links below)
On image:
``` shell
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
```

On video:
``` shell
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source yourvideo.mp4
```

### GradCAM on Yolov7 
On images:
``` shell
python main_gradcam.py --model-path weights/yolov7.pt --conf 0.75 --img-size 640 --img-path inference/images/horses.jpg --method gradcam --target-layers 104_act
```

**Notes:** 
- See main_gradcam.py for detailed running options (also in illustrations below)
- Saliency maps will be saved to outputs/horses/<method> for each class detected.

On video: 
``` shell
TODO
```

## Project presentation

![Alt Text](figure/XAI_Yolov7_GradCam_highRes.gif)

## Acknowledgements

* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
* [https://github.com/pooya-mohammadi/yolov5-gradcam](https://github.com/pooya-mohammadi/yolov5-gradcam)
* [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
