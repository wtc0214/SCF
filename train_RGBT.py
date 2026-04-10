import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R'E:\YOLOv11-RGBT-master\YOLOv11-RGBT-master\ultralytics\cfg\models\11-RGBT\yolo11-RGBRGB6C-midfusion.yaml')
    # model.info(True,True)
    model.load('E:/YOLOv11-RGBT-master/YOLOv11-RGBT-master/runs/M3FD(SCF)/yolo11s-RGBT-midfusion-all3/weights/best.pt') # loading pretrain weights
    # "E:/YOLOv11-RGBT-master/YOLOv11-RGBT-master/ultralytics/cfg/datasets/KAIST.yaml"
    model.train(data="E:/YOLOv11-RGBT-master/YOLOv11-RGBT-master/ultralytics/cfg/datasets/M3FD.yaml",
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=10,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                patience=20,
                # lr0=0.002,
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGBT",
                channels=4,
                project='runs/m3fd(heads)',
                name='yolo11s-RGBT-midfusion-all',
                )