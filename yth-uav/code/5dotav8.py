import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,RTDETRyh1,RTDETRpyh1,YOLO

if __name__ == '__main__':

    model = YOLO('ultralytics/cfg/models/rt-detr/llf/vis/newdota.yaml')
    model.load(weights='/home/cv/lyh/lyhrtdetr2/runs/dota/v8_/weights/best.pt')# loading pretrain weights
    model.train(data='/home/cv/lyh/lyhrtdetr2/dataset/dota.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=4,
                device=1,
                 # last.pt path

                project='runs/dota',
                name='newdota_',
                )
