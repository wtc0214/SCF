import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('E:/YOLOv11-RGBT-master/YOLOv11-RGBT-master/ultralytics/cfg/models/11/yolo11.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=R'E:/YOLOv11-RGBT-master/YOLOv11-RGBT-master/ultralytics/cfg/datasets/LLVIP.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=10,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="Gray",  # Gray16bit
                channels=1,
                project='Drone_if',
                name='BCCD-yolov11n-',
                )