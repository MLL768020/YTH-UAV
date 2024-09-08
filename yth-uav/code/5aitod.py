import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,RTDETRyh1,RTDETRpyh1,YOLO,YOLO1

if __name__ == '__main__':

    model = YOLO1('/home/cv/lyh/lyhrtdetr2/ultralytics/cfg/models/rt-detr/llf/vis/node.yaml')
    model.load(weights='/home/cv/lyh/lyhrtdetr2/weights/yolov8l.pt')# loading pretrain weights
    model.train(data='/home/cv/lyh/lyhrtdetr2/dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                workers=6,
                device=2,
                resume='/home/cv/lyh/lyhrtdetr2/runs/vis/node2/weights/last.pt', # last.pt path
                project='runs/vis',
                name='node',
                )
