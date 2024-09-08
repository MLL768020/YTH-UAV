import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':

    model = RTDETR('ultralytics/cfg/models/rt-detr/llf/vis/afpn.yaml')
    model.load(weights='/home/liyihang/lyhredetr/weights/yolov8l.pt')# loading pretrain weights
    model.train(data='/home/liyihang/lyhredetr/dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=1,
                device=4,
                resume='/home/liyihang/lyhredetr/runs/vis/afpn_/weights/last.pt', # last.pt path
                project='/home/liyihang/lyhredetr/runs/vis',
                name='afpn_',
                )
