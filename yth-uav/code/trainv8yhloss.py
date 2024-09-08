import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,YOLO

if __name__ == '__main__':

    model = YOLO('ultralytics/cfg/models/rt-detr/llf/vis/yolov8vd.yaml')
    model.load(weights='/home/liyihang/lyhredetr/weights/yolov8l.pt')# loading pretrain weights
    model.train(data='/home/liyihang/lyhredetr/dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=2,
                device=5,
                #resume='/home/liyihang/lyhredetr/runs/vis/fpn_10/weights/last.pt', # last.pt path
                project='/home/liyihang/lyhredetr/runs/vis',
                name='v8yhloss_',
                )
