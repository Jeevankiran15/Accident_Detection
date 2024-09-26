from ultralytics import YOLO

def main():
    model = YOLO('yolov8m.pt')
    results = model.train(data='AccidentsDetectionYOLOv8/data1.yaml', epochs=20, imgsz=640,batch=16)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
