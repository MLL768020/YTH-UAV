import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,RTDETRyh1,RTDETRpyh1,YOLO

if __name__ == '__main__':

    model = YOLO('/home/cv/lyh/lyhrtdetr2/ultralytics/cfg/models/rt-detr/llf/vis/node.yaml')
    model.load(weights='/home/cv/lyh/lyhrtdetr2/runs/pheno/val2/weights/best.pt')# loading pretrain weights
    model.train(data='/home/cv/lyh/lyhrtdetr2/dataset/pheno.yaml',
                cache=False,
                imgsz=640,
                epochs=12,
                batch=4,
                workers=2,
                device=2,
                #resume='/home/cv/lyh/lyhrtdetr2/runs/pheno/yolo/weights/last.pt', # last.pt path
                project='runs/pheno',
                name='pheno',
                )
