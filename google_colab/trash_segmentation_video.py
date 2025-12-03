import cv2
from ultralytics import YOLO

model = YOLO('runs/detect/yolov8m_entrenamiento8/weights/best.pt')
cap = cv2.VideoCapture('Service Project/images/video/Trash_Segmentation.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    results = model.track(frame, persist=True)
    annotated = results[0].plot()  
    cv2.imshow('Detección de basura', annotated)
    if cv2.waitKey(1) == ord('q'): break
cap.release()
cv2.destroyAllWindows()
for res in results:           
    img_annot = res.plot()      
    cv2.imshow('Detección', img_annot)
    cv2.waitKey(0)
