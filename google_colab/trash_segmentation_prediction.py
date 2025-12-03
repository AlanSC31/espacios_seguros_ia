from ultralytics import YOLO
model = YOLO('runs/detect/yolov8m_entrenamiento8/weights/best.pt')
results = model(
    'Service Project/images/prediction/predictionImage.jpg', 
    conf=0.25)