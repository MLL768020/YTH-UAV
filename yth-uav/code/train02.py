import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':

    model = RTDETR('ultralytics/cfg/models/rt-detr/llf/vis/v8rtdetr/DAV.yaml')
    model.load(weights='/home/liyihang/lyhredetr/weights/yolov8l.pt')# loading pretrain weights
    model.train(data='/home/liyihang/lyhredetr/dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                workers=2,
                device=0,
                # resume='/home/liyihang/lyhredetr/runs/llf/yoloredetraa/weights/last.pt', # last.pt path
                project='/home/liyihang/lyhredetr/runs/ai',
                name='train02_',
                )
