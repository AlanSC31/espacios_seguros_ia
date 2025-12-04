import cv2
import time
import numpy as np
import mysql.connector
import requests
import supervision as sv
from ultralytics import YOLO
from roboflow import Roboflow
from collections import defaultdict, deque
from datetime import datetime
from google.colab.patches import cv2_imshow
from IPython.display import clear_output


IMGBB_API_KEY = "a5e5364eafbc5ed33d8f616904e2d797" 

DB_CONFIG_ARMAS = {
    'host': 'sql5.freesqldatabase.com',
    'user': 'sql5809114',
    'password': 'mFNareUBz5', 
    'database': 'sql5809114',
    'port': 3306
}

DB_CONFIG_AGRESIONES = {
    'host': 'sql5.freesqldatabase.com',
    'user': 'sql5809114',
    'password': 'mFNareUBz5', 
    'database': 'sql5809114',
    'port': 3306
}

VIDEO_SOURCE = "video3.mp4"
OUTPUT_NAME = "video_final_precision.mp4"


def subir_a_imgbb(frame, etiqueta):
    try:
        filename = f"temp_{etiqueta}.jpg"
        cv2.imwrite(filename, frame)
        url = "https://api.imgbb.com/1/upload"
        payload = {
            "key": IMGBB_API_KEY,
            "name": f"alerta_{etiqueta}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        with open(filename, "rb") as file:
            files = {"image": file}
            response = requests.post(url, data=payload, files=files)
        if response.status_code == 200:
            return response.json()["data"]["url"]
        return None
    except Exception:
        return None

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_torso_scale(pose):
    """Calcula la altura del torso (cuello a cadera) para normalizar distancias"""
    
    shoulder_mid = (np.array(pose[5]) + np.array(pose[6])) / 2
    hip_mid = (np.array(pose[11]) + np.array(pose[12])) / 2
    scale = np.linalg.norm(shoulder_mid - hip_mid)
    
    return scale if scale > 0 else 1.0

def hand_speed(history):
    if len(history) < 2: return 0.0
    speeds = [euclidean(history[i], history[i+1]) for i in range(len(history)-1)]
    return np.mean(speeds)

def detect_aggression_normalized(pose1, pose2, speed_h1, speed_h2):
    """
    Detecta agresion basandose en la escala del cuerpo, no en pixeles fijos.
    """
    scale1 = get_torso_scale(pose1)
    scale2 = get_torso_scale(pose2)
    avg_scale = (scale1 + scale2) / 2
    
    center1 = pose1[0] # Nariz como referencia rapida o pecho
    center2 = pose2[0]
    dist_abs = euclidean(center1, center2)
    
    dist_norm = dist_abs / avg_scale
    esta_cerca = dist_norm < 2.5 

    if not esta_cerca:
        return False

    speed_norm1 = speed_h1 / avg_scale
    speed_norm2 = speed_h2 / avg_scale
    
    golpe = (speed_norm1 > 0.15) or (speed_norm2 > 0.15)
    
    pelea = (speed_norm1 > 0.10) and (speed_norm2 > 0.10)
    
    guardia1 = pose1[9][1] < pose1[5][1] or pose1[10][1] < pose1[6][1]
    guardia2 = pose2[9][1] < pose2[5][1] or pose2[10][1] < pose2[6][1]

    return (golpe or pelea) and (esta_cerca or guardia1 or guardia2)


def insert_weapon_db(track_id, class_name, image_url):
    tipo = "arma_blanca" if str(class_name) == "0" else "arma_fuego"
    try:
        conn = mysql.connector.connect(**DB_CONFIG_ARMAS)
        cursor = conn.cursor()
        query = "INSERT INTO armas_deteccion (id_persona, tipo_arma) VALUES (%s, %s)"
        cursor.execute(query, (int(track_id), tipo))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[BD ARMAS] OK ID: {track_id}")
    except Exception as e:
        pass 

def insert_aggression_db(frame_num, num_people, image_url):
    try:
        conn = mysql.connector.connect(**DB_CONFIG_AGRESIONES)
        cursor = conn.cursor()
        sql = "INSERT INTO agresiones_detectadas (timestamp, frame_num, cant_personas, tipo_agresion) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (datetime.now(), int(frame_num), int(num_people), "fisica_detectada"))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[BD AGRESION] OK Frame: {frame_num}")
    except Exception as e:
        pass


print("Cargando Roboflow...")
rf = Roboflow(api_key="5b3nLnlY9wMoNfAcQ51R")
project = rf.workspace("m-qczea").project("weapon_detection_v2-rdpq3")
model_weapons = project.version(2).model

print("Cargando YOLO Pose...")
model_pose = YOLO("yolov8n-pose.pt")


tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator(thickness=4) # Grosor 4 para que se vea bien
label_annotator = sv.LabelAnnotator(text_scale=0.8, text_thickness=2)

hand_histories = defaultdict(lambda: deque(maxlen=6)) # Historial corto para respuesta rapida

cap = cv2.VideoCapture(VIDEO_SOURCE)
width, height = int(cap.get(3)), int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(OUTPUT_NAME, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0
last_weapon_time = 0
last_aggression_time = 0
COOLDOWN = 2.0 # Segundos entre alertas para no saturar

print("--- INICIANDO PROCESAMIENTO DE ALTA PRECISION ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1
    current_time = time.time()
    
    try:
        res_w = model_weapons.predict(frame, confidence=25, overlap=50).json()
        detections = sv.Detections.from_inference(res_w)
        
        if len(detections) > 0:
            detections = tracker.update_with_detections(detections)
            
            frame = box_annotator.annotate(scene=frame, detections=detections)
            
            if detections.tracker_id is not None:
                labels = [f"#{t_id} {n} {c:.2f}" for t_id, n, c in zip(detections.tracker_id, detections['class_name'], detections.confidence)]
                frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
                
                if (current_time - last_weapon_time) > COOLDOWN:
                    print(f"!!! ARMA DETECTADA !!!")
                    url = subir_a_imgbb(frame, "arma")
                    # Tomamos el primer ID detectado para la BD
                    insert_weapon_db(detections.tracker_id[0], detections['class_name'][0], url)
                    last_weapon_time = current_time

    except Exception: pass

    try:
        results_pose = model_pose.predict(frame, conf=0.5, verbose=False)[0]
        keypoints_all = []
        boxes_p = []

        if results_pose.boxes:
            for i, kp in enumerate(results_pose.keypoints):
                pts = kp.data[0].cpu().numpy() # x, y, conf
                if len(pts) > 0:
                    kps_list = [(int(x), int(y)) for x, y, c in pts]
                    keypoints_all.append(kps_list)
                    box = results_pose.boxes[i].xyxy[0].cpu().numpy().astype(int)
                    boxes_p.append(box)
                    
                    pid = tuple(box)
                    hand_histories[pid].append(kps_list[9]) # Muñeca derecha

            agresion_frame = False
            for i in range(len(keypoints_all)):
                for j in range(i + 1, len(keypoints_all)):
                    
                    id_i, id_j = tuple(boxes_p[i]), tuple(boxes_p[j])
                    speed_i = hand_speed(hand_histories[id_i])
                    speed_j = hand_speed(hand_histories[id_j])
                    
                    if detect_aggression_normalized(keypoints_all[i], keypoints_all[j], speed_i, speed_j):
                        agresion_frame = True
                        
                        c1 = keypoints_all[i][0] # Nariz
                        c2 = keypoints_all[j][0]
                        cv2.line(frame, c1, c2, (0, 0, 255), 4)
                        
                        # Cajas rojas alrededor de ambos
                        bx1 = boxes_p[i]
                        bx2 = boxes_p[j]
                        cv2.rectangle(frame, (bx1[0], bx1[1]), (bx1[2], bx1[3]), (0, 0, 255), 4)
                        cv2.rectangle(frame, (bx2[0], bx2[1]), (bx2[2], bx2[3]), (0, 0, 255), 4)
                        cv2.putText(frame, "AGRESION DETECTADA", (bx1[0], bx1[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

            if agresion_frame and (current_time - last_aggression_time) > COOLDOWN:
                print("!!! AGRESION DETECTADA !!!")
                url = subir_a_imgbb(frame, "agresion")
                insert_aggression_db(frame_count, len(keypoints_all), url)
                last_aggression_time = current_time

    except Exception as e: pass

    out.write(frame)
    if frame_count % 10 == 0:
        clear_output(wait=True)
        print(f"Procesando Frame: {frame_count}")
        cv2_imshow(cv2.resize(frame, (480, 270)))

cap.release()
out.release()
print(f"Finalizado. Descarga: {OUTPUT_NAME}")