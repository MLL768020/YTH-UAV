import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,RTDETRyh1,RTDETRpyh1,YOLO

if __name__ == '__main__':

    model = RTDETRyh1('/home/cv/lyh/lyhrtdetr2/ultralytics/cfg/models/rt-detr/llf/vis/n.yaml')
    model.load(weights='/home/cv/lyh/lyhrtdetr2/weights/yolov8n.pt')# loading pretrain weights
    model.train(data='/home/cv/lyh/lyhrtdetr2/dataset/pheno.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                workers=4,
                device=2,
                #resume='/home/cv/lyh/lyhrtdetr2/runs/pheno/pheno3/weights/best.pt', # last.pt path
                project='runs/pheno',
                name='spheno',
                )
