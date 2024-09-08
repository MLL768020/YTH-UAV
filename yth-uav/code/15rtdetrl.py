import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':

    model = RTDETR('ultralytics/cfg/models/rt-detr/s-rtdetr-l.yaml')
    model.load(weights='/home/liyihang/lyhredetr/weights/rtdetr-l.pt')  # loading pretrain weights
    model.train(data='dataset/dota.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=4,
                device=2,
                # resume=True, # last.pt path
                project='runs/llf',
                name='15rtdetrl',
                )
