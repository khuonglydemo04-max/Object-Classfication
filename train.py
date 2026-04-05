from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')
model.train(data='mango_dataset', epochs=35, imgsz=224)
