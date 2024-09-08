import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('/home/liyihang/lyhredetr/runs/vis/YTH-UAV/weights/best.pt')  # select your model.pt path
    model.predict(source='/home/liyihang/miji/uav_palm_data/val/images/',
                  project='runs/show',
                  name='RTYO2',
                  save=True,
                  visualize=False,  # visualize model features maps

                  )
