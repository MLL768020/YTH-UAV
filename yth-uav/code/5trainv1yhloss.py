import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,RTDETRyh1,RTDETRpyh1

if __name__ == '__main__':

    model = RTDETRyh1('ultralytics/cfg/models/rt-detr/llf/vis/fpnupdate.yaml')
    model.load(weights='/home/cv/lyh/lyhrtdetr2/weights/yolov8l.pt')# loading pretrain weights
    model.train(data='/home/cv/lyh/lyhrtdetr2/dataset/pheno.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=4,
                device=1,
                #resume='/home/liyihang/lyhredetr/runs/vis/v1yhloss_3/weights/last.pt', # last.pt path
                project='runs/pheno',
                name='v1yhloss_',
                )
