import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

# RTDETR giou
if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/llf/vis/rtdetr-r50.yaml')
    model.load(weights='/home/liyihang/lyhredetr/weights/rtdetr-r50.pt')  # loading pretrain weights
    model.train(data='/home/liyihang/lyhredetr/dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=1,
                device=2,
                # resume ='/home/liyihang/lyhredetr/runs/vis/yuan_3/weights/last.pt',
                project='/home/liyihang/lyhredetr/runs/vis',
                name='r50vis_',
                )
