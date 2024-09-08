import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,RTDETRyh1

if __name__ == '__main__':

    model = RTDETRyh1('ultralytics/cfg/models/rt-detr/llf/vis/fpnupdate.yaml')
    model.load(weights='/home/liyihang/lyhredetr/weights/v1loss1best.pt')# loading pretrain weights
    model.train(data='/home/liyihang/lyhredetr/dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=30,
                batch=4,
                workers=4,
                device=4,
                #resume='/home/liyihang/lyhredetr/runs/vis/v1yhloss_/weights/last.pt', # last.pt path
                project='/home/liyihang/lyhredetr/runs/vis',
                name='trainv1loss1_',
                )
