import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,RTDETRyh1,RTDETR1

if __name__ == '__main__':

    model = RTDETR1('ultralytics/cfg/models/rt-detr/llf/vis/fpnupdate.yaml')
    model.load(weights='/home/liyihang/lyhredetr/weights/yolov8l.pt')# loading pretrain weights
    model.train(data='/home/liyihang/lyhredetr/dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=2,
                device=6,
                #resume='/home/liyihang/lyhredetr/runs/vis/v1yhloss_/weights/last.pt', # last.pt path
                project='/home/liyihang/lyhredetr/runs/vis',
                name='v1gyhloss_',
                )
