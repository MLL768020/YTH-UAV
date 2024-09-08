import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,YOLO

if __name__ == '__main__':

    model = YOLO('ultralytics/cfg/models/rt-detr/llf/vis/yolov8vd.yaml')
    #model.load(weights='/home/liyihang/lyhredetr/weights/yolov8l.pt')# loading pretrain weights
    model.train(data='/home/cv/Project/yolov7-main/data/quexian.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=2,
                device='3',
                #resume='/home/liyihang/lyhredetr/runs/vis/fpn_10/weights/last.pt', # last.pt path
                project='runs/quexian',
                name='v8_',
                )
