import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':

    model = RTDETR('/home/liyihang/lyhredetr/ultralytics/cfg/models/rt-detr/llf/ai/v8rtdetr/yuan.yaml')
    model.load(weights='/home/liyihang/lyhredetr/weights/yolov8l.pt')  # loading pretrain weights
    model.train(data='/home/liyihang/lyhredetr/dataset/uavnc1.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=4,
                device=2,
                # resume='/home/liyihang/lyhredetr/runs/llf/yoloredetraa/weights/last.pt', # last.pt path
                project='/home/liyihang/lyhredetr/runs/uavnc1_',
                name='train01_',
                )
