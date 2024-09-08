import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':

    model = RTDETR('ultralytics/cfg/models/rt-detr/llf/dota/v8rtdetr/yuan.yaml')
    model.load(weights='/home/liyihang/lyhredetr/weights/yolov8l.pt')  # loading pretrain weights
    model.train(data='dataset/dota.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=1,
                device=4,
                # resume=True, # last.pt path
                project='runs/dota',
                name='15yuan',
                )
