import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO,RTDETR

if __name__ == '__main__':
    model = RTDETR('/home/cv/lyh/lyhrtdetr2/runs/ai-tod/best.pt')
    model.val(data='dataset/AI-TOD.yaml',
              split='val',
              imgsz=640,
              batch=1,

              save_json=True,  # if you need to cal coco metrice
              device=1,
              project='runs/val',
              name='ta',
              )
