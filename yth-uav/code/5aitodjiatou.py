import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,RTDETRyh1,RTDETRpyh1,YOLO

if __name__ == '__main__':

    model = RTDETRyh1('/home/cv/lyh/lyhrtdetr2/ultralytics/cfg/models/rt-detr/llf/vis/483.yaml')
    model.load(weights='/home/cv/lyh/lyhrtdetr2/weights/yolov8s.pt')# loading pretrain weights
    model.train(data='/home/cv/lyh/lyhrtdetr2/dataset/AI-TOD.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=4,
                device='2',
                #resume='/home/liyihang/lyhredetr/runs/vis/v1yhloss_3/weights/last.pt', # last.pt path
                project='runs/ai-tod',
                name='aitodjiatou_',
                )
