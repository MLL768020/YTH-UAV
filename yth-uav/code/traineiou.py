import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,RTDETRyh1,RTDETR2
# RTDETR giou
if __name__ == '__main__':

    model = RTDETR2('ultralytics/cfg/models/rt-detr/llf/vis/yolov8256.yaml')
    model.load(weights='/home/liyihang/lyhredetr/weights/yolov8l.pt')  # loading pretrain weights
    model.train(data='/home/liyihang/lyhredetr/dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=1,
                device=3,
                #resume ='/home/liyihang/lyhredetr/runs/vis/v8256_/weights/last.pt',
                project='/home/liyihang/lyhredetr/runs/vis',
                name='256eiou_',
                )
