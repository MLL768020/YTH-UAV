import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR1

if __name__ == '__main__':

    model = RTDETR1('ultralytics/cfg/models/rt-detr/llf/nc/lv8rtdetrnew07.yaml')
    model.load(weights='/home/liyihang/lyhredetr/weights/yolov8l.pt')  # loading pretrain weights
    model.train(data='dataset/dota.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=4,
                device=1,
                # resume=True, # last.pt path
                project='runs/llf',
                name='15self',
                )
