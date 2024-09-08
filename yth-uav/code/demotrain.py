import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/llf/v8rtdetr.yaml')
    model.load(weights='/home/cv/lyh/lyhredetr/weights/yolov8l.pt')  # loading pretrain weights
    model.train(data='dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=8,
                device=1,
                # resume=True, # last.pt path
                project='runs/llf',
                name='demo5train',
                )
