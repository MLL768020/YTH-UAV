import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,RTDETRyh1,RTDETRpyh1,YOLO

if __name__ == '__main__':

    model = YOLO('/home/cv/lyh/lyhrtdetr2/ultralytics/cfg/models/rt-detr/llf/vis/yolo.yaml')
    model.load(weights='/home/cv/lyh/lyhrtdetr2/runs/ai-tod/aitod23/weights/last.pt')# loading pretrain weights
    model.train(data='/home/cv/lyh/lyhrtdetr2/dataset/AI-TOD.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                workers=4,
                device='2',
                #resume='/home/cv/lyh/lyhrtdetr2/runs/ai-tod/aitod23/weights/last.pt', # last.pt path
                project='runs/ai-tod',
                name='aitod',
                )
