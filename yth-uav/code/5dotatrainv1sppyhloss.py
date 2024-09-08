import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,RTDETRyh1,RTDETRpyh1

if __name__ == '__main__':

    model = RTDETRyh1('ultralytics/cfg/models/rt-detr/llf/dota/v1spp.yaml')
    model.load(weights='/home/cv/lyh/lyhrtdetr2/weights/yolov8l.pt')# loading pretrain weights
    model.train(data='/home/cv/lyh/lyhrtdetr2/dataset/dota.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=4,
                device='2',
                #resume='', # last.pt path
                project='runs/dota',
                name='v1sppyhloss_',
                )
