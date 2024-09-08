import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,RTDETRyh1,RTDETRpyh1
# RTDETR giou
if __name__ == '__main__':

    model = RTDETRpyh1('ultralytics/cfg/models/rt-detr/llf/vis/fpnupdate.yaml')
    model.load(weights='/home/liyihang/lyhredetr/weights/yolov8l.pt')  # loading pretrain weights
    model.train(data='/home/liyihang/lyhredetr/dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                workers=8,
                device=6,
                #resume ='/home/liyihang/lyhredetr/runs/vis/v8256_/weights/last.pt',
                project='/home/liyihang/lyhredetr/runs/vis',
                name='ppyhlossv1_',
                )
