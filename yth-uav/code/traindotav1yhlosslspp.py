import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,RTDETRyh1
# RTDETR giou
if __name__ == '__main__':

    model = RTDETRyh1('ultralytics/cfg/models/rt-detr/llf/vis/s.yaml')
    model.load(weights='/home/cv/lyh/lyhrtdetr2/runs/ai-tod/best.pt')  # loading pretrain weights
    model.train(data='/home/cv/lyh/lyhrtdetr2/dataset/AI-TOD.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                workers=4,
                device='1,2,3,4',
                #resume ='/home/cv/lyh/lyhrtdetr2/runs/ai-tod/aitodlspp_/weights/last.pt',
                project='runs/aitod',
                name='v1yhlosslspp_',
                )
