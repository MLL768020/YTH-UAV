import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,RTDETR2
# RTDETR giou
if __name__ == '__main__':

    model = RTDETR2('ultralytics/cfg/models/rt-detr/llf/vis/yolov8256.yaml')
    model.load(weights='/home/liyihang/lyhredetr/weights/yolov8l.pt')  # loading pretrain weights
    model.train(data='/home/liyihang/lyhredetr/dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=2,
                workers=1,
                device=2,
                #resume ='/home/liyihang/lyhredetr/runs/vis/ciouv8256_/weights/last.pt',
                project='/home/liyihang/lyhredetr/runs/vis',
                name='2ciouv8256_',
                )
