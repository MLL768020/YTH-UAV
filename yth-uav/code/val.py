import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,YOLO

if __name__ == '__main__':
    model = YOLO('/home/cv/lyh/lyhrtdetr2/runs/ai-tod/v8_7/weights/last.pt')
    model.val(data='dataset/AI-TOD.yaml',
              split='val',
              imgsz=640,
              batch=1,

              save_json=True,  # if you need to cal coco metrice
              device=0,
              project='runs/val',
              name='ta',
              )
