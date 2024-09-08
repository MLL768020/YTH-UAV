import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':

    model = RTDETR('/home/liyihang/lyhredetr/ultralytics/cfg/models/rt-detr/llf/ai/v8rtdetr/DAV.yaml')
    model.load(weights='/home/liyihang/lyhredetr/weights/yolov8l.pt')  # loading pretrain weights
    model.train(data='/home/liyihang/lyhredetr/dataset/dota.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=2,
                workers=1,
                device=1,
                # resume='/home/liyihang/lyhredetr/runs/llf/yoloredetraa/weights/last.pt', # last.pt path
                project='/home/liyihang/lyhredetr/runs/ai',
                name='train02_',
                )
