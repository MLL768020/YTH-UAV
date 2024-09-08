import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETRyh1

# RTDETR giou
if __name__ == '__main__':

    model = RTDETRyh1('ultralytics/cfg/models/rt-detr/llf/vis/node.yaml')
    model.load(weights='/home/cv/lyh/lyhrtdetr2/weights/yolov8l.pt')  # loading pretrain weights
    model.train(data='dataset/dota.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=4,
                device='1,2,3,4',
                #resume ='/home/cv/lyh/lyhrtdetr2/runs/vis/aitod_9/weights/last.pt',
                project='runs/dota',
                name='node_',
                )
