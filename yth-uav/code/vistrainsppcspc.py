import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR, RTDETRyh1

# RTDETR giou
if __name__ == '__main__':
    model = RTDETRyh1('ultralytics/cfg/models/rt-detr/llf/vis/sppcspc.yaml')
    model.load(weights='weights/v8256.pt')  # loading pretrain weights
    model.train(data='dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=2,
                device=4,
                #resume ='',
                project='runs/vis',
                name='sppcspc_',
                )
