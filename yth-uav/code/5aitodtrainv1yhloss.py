import warnings

warnings.filterwarnings('ignore')

from ultralytics import RTDETR,RTDETRyh1

if __name__ == '__main__':

    model = RTDETRyh1('ultralytics/cfg/models/rt-detr/llf/vis/fpnupdate.yaml')
    model.load(weights='/home/cv/lyh/lyhrtdetr2/weights/yolov8l.pt')# loading pretrain weights
    model.train(data='/home/cv/lyh/lyhrtdetr2/dataset/AI-TOD.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=1,
                workers=1,
                device='4',
                #resume='/home/cv/lyh/lyhrtdetr2/runs/ai-tod/v1yhloss_/weights/last.pt', # last.pt path
                project='/home/cv/lyh/lyhrtdetr2/runs/ai-tod',
                name='aitod_',
                )
