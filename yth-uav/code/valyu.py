import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':

    model = RTDETR('ultralytics/cfg/models/rt-detr/llf/nc/lv8rtdetr.yaml')
    model.load(weights='/home/liyihang/lyhredetr/weights/yolov8l.pt')  # loading pretrain weights
    model.train(data='dataset/uav_palm_data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=4,
                device=1,
                # resume=True, # last.pt path
                project='runs/llf',
                name='ncv8lrtdetryuan',
                )
