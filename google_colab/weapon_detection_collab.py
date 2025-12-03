import cv2
import time
import mysql.connector
import supervision as sv
from roboflow import Roboflow
from google.colab.patches import cv2_imshow
from IPython.display import clear_output 


DB_CONFIG = {
    'host': 'sql5.freesqldatabase.com',
    'user': 'sql5809114',
    'password': 'mFNareUBz5',
    'database': 'sql5809114',
    'port': 3306
}

#Modelo Roboflow
rf = Roboflow(api_key="5b3nLnlY9wMoNfAcQ51R")
project = rf.workspace("m-qczea").project("weapon_detection_v2-rdpq3")
version = project.version(2)
model = version.model


VIDEO_SOURCE = "video3.mp4" 
OUTPUT_NAME = "video_procesado_final.mp4"
CONFIDENCE_THRESHOLD = 50 
IOU_THRESHOLD = 0.5        
COOLDOWN_SECONDS = 60.0  # Tiempo mínimo entre alertas (1 minuto)

last_global_alert_time = 0 

def insert_into_db(track_id, class_name):
   
    global last_global_alert_time 
    current_time = time.time()
    
    tipo_arma =  str(class_name)
    if (tipo_arma == "0"):
        tipo_arma = "arma_blanca"
    elif (tipo_arma == "1"):
        tipo_arma = "arma_fuego"
    

    time_since_last = current_time - last_global_alert_time
    if time_since_last < COOLDOWN_SECONDS:
        print(f" Cooldown activo. Ignorando ID {track_id}. Faltan {int(COOLDOWN_SECONDS - time_since_last)}s")
        return 

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        query = "INSERT INTO armas_deteccion (id_persona, tipo_arma) VALUES (%s, %s)"
        cursor.execute(query, (track_id, str(tipo_arma)))
        conn.commit()
        
        print(f" [ALERTA BD GUARDADA] ID: {track_id} | Tipo: {tipo_arma}")
        
        last_global_alert_time = current_time 
        
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        print(f" Error SQL: {err}")


tracker = sv.ByteTrack() 
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(VIDEO_SOURCE)


if cap.isOpened():
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(OUTPUT_NAME, fourcc, fps, (width, height))
    print("Iniciando análisis...")
else:
    print("Error al abrir video.")


frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Inferencia
    try:
        results = model.predict(frame, confidence=CONFIDENCE_THRESHOLD, overlap=IOU_THRESHOLD).json()
        detections = sv.Detections.from_inference(results)
    except Exception as e:
        print(f"Error API: {e}")
        continue

    if len(detections) > 0:
        # Tracking
        detections = tracker.update_with_detections(detections)
        
        tracker_ids = detections.tracker_id
        confidences = detections.confidence
        class_names = detections['class_name']
        
        if (tracker_ids is not None) and (confidences is not None) and (class_names is not None):
            
            labels = []
            for tracker_id, class_name, confidence in zip(tracker_ids, class_names, confidences):
                
            
                labels.append(f"ID:{tracker_id} {class_name} {confidence:.2f}")
                
                insert_into_db(track_id=int(tracker_id), class_name=class_name)

            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    video_writer.write(frame)

    # Mostrar progreso cada 10 frames
    if frame_count % 10 == 0: 
        clear_output(wait=True)
        print(f"Procesando frame: {frame_count} | Última alerta hace: {int(time.time() - last_global_alert_time)}s")
        cv2_imshow(cv2.resize(frame, (640, 360)))

cap.release()
video_writer.release()
print(f"Finalizado. Video guardado: {OUTPUT_NAME}")