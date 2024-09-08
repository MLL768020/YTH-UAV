import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,RTDETRyh1,RTDETRpyh1

if __name__ == '__main__':

    model = RTDETRyh1('ultralytics/cfg/models/rt-detr/llf/vis/lsppv1.yaml')
    model.load(weights='/home/cv/lyh/lyhrtdetr2/runs/dota/dota16/weights/last.pt')# loading pretrain weights
    model.train(data='/home/cv/lyh/lyhrtdetr2/dataset/dota.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=4,
                device=2,
                 # last.pt path

                project='runs/dota',
                name='5dotaval',
                )
