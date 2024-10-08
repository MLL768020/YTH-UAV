import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,RTDETRyh1
# RTDETR giou
if __name__ == '__main__':

    model = RTDETRyh1('ultralytics/cfg/models/rt-detr/llf/vis/yolov8256lspp.yaml')
    model.load(weights='/home/cv/lyh/lyhrtdetr2/weights/v8256.pt')  # loading pretrain weights
    model.train(data='dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=4,
                device=0,
                #resume ='/home/liyihang/lyhredetr/runs/vis/yuan_3/weights/last.pt',
                project='runs/vis',
                name='lspp_',
                )
