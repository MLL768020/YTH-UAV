import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,RTDETRyh1,RTDETRpyh1,YOLO

if __name__ == '__main__':

    model = YOLO('ultralytics/cfg/models/rt-detr/llf/vis/yolov8vd.yaml')
    model.load(weights='/home/cv/lyh/lyhrtdetr2/weights/yolov8s.pt')# loading pretrain weights
    model.train(data='/home/cv/lyh/lyhrtdetr2/dataset/pheno.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=2,
                workers=1,
                device=2,
                #resume='/home/liyihang/lyhredetr/runs/vis/v1yhloss_3/weights/last.pt', # last.pt path
                project='runs/pheno',
                name='sphenov8',
                )
