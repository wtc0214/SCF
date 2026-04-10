import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R'E:\YOLOv11-RGBT-master\YOLOv11-RGBT-master\ultralytics\cfg\models\v8\yolov8_inr_enhanced.yaml')
    model.load('E:/YOLOv11-RGBT-master/YOLOv11-RGBT-master/best.pt') # loading pretrain weights
    model.train(data="E:/YOLOv11-RGBT-master/YOLOv11-RGBT-master/ultralytics/cfg/datasets/UAVDT.yaml",
                cache=False,
                imgsz=640,
                epochs=900,
                batch=32,
                close_mosaic=5,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                patience=20,
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGB",
                channels=3,
                project='COCO_aarm_',
                name='BCCD-ppyoloe-s',
                )