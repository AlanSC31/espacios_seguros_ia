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

COOLDOWN_DB = 2.0    
COOLDOWN_IMG = 30.0  

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

VIDEO_SOURCE = "gym_attack.mp4"
OUTPUT_NAME = "resultado_cctv_final.mp4"


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
    except: pass

def insert_aggression_db(frame_num, num_people, tipo, image_url):
    try:
        conn = mysql.connector.connect(**DB_CONFIG_AGRESIONES)
        cursor = conn.cursor()
        sql = "INSERT INTO agresiones_detectadas (timestamp, frame_num, cant_personas, tipo_agresion) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (datetime.now(), int(frame_num), int(num_people), tipo))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[BD AGRESION] OK Frame: {frame_num} Tipo: {tipo}")
    except: pass


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def draw_skeleton(frame, kps):
    limbs = [(5,7), (7,9), (6,8), (8,10), (5,6), (5,11), (6,12), (11,12)]
    for p1, p2 in limbs:
        if p1 < len(kps) and p2 < len(kps):
            pt1 = kps[p1]
            pt2 = kps[p2]
            if pt1[0] > 0 and pt2[0] > 0:
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
    return frame

def get_cctv_scale(pose):
    shoulder_dist = euclidean(pose[5], pose[6])
    if shoulder_dist < 10: return 40.0 # Valor por defecto si falla deteccion
    return shoulder_dist

def get_centroid(pose):
    return ((np.array(pose[5]) + np.array(pose[6])) / 2).astype(int)

def hand_speed(history):
    if len(history) < 2: return 0.0
    dist_total = 0
    for i in range(len(history)-1):
        dist_total += euclidean(history[i], history[i+1])
    return dist_total / (len(history)-1)

def detect_aggression_cctv(pose1, pose2, speed_h1, speed_h2):
    scale = (get_cctv_scale(pose1) + get_cctv_scale(pose2)) / 2
    
    center1 = get_centroid(pose1)
    center2 = get_centroid(pose2)
    dist_real = euclidean(center1, center2)
    
    esta_en_rango = dist_real < (scale * 3.5) 

    if not esta_en_rango:
        return False, ""

    speed_norm1 = speed_h1 / scale
    speed_norm2 = speed_h2 / scale
    
    movimiento_rapido = (speed_norm1 > 0.3) or (speed_norm2 > 0.3)
    
    wrist_r1, wrist_l1 = pose1[9], pose1[10]
    head_target = center2
    dist_punch_1 = min(euclidean(wrist_r1, head_target), euclidean(wrist_l1, head_target))
    contacto_fisico = dist_punch_1 < (scale * 1.5)
    
    # REGLAS
    if movimiento_rapido and contacto_fisico:
        return True, "GOLPE_CERCANO"
    if movimiento_rapido and esta_en_rango:
        return True, "MOVIMIENTO_BRUSCO"

    return False, ""


print("Cargando Roboflow (Armas)...")
rf = Roboflow(api_key="5b3nLnlY9wMoNfAcQ51R")
project = rf.workspace("m-qczea").project("weapon_detection_v2-rdpq3")
model_weapons = project.version(2).model

print("Cargando YOLO Pose (Medium)...")
try:
    # Intenta cargar Medium para precision. Si falla memoria, usa Nano
    model_pose = YOLO("yolov8m-pose.pt") 
except:
    print("Medium fallo, usando Nano...")
    model_pose = YOLO("yolov8n-pose.pt")

# --- 5. BUCLE PRINCIPAL ---

tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.DEFAULT, thickness=4)
label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=2)

hand_histories = defaultdict(lambda: deque(maxlen=8)) # Historial un poco mas largo para CCTV

cap = cv2.VideoCapture(VIDEO_SOURCE)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(OUTPUT_NAME, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0
last_w_db, last_w_img = 0, 0
last_a_db, last_a_img = 0, 0

print(f"--- PROCESANDO VIDEO (MODO CCTV) ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1
    current_time = time.time()
    
    # --- MODULO A: ARMAS (Alta confianza para evitar mochilas) ---
    try:
        # Confidence 45% filtra la mayoria de objetos basura en gimnasios
        res_w = model_weapons.predict(frame, confidence=45, overlap=50).json()
        detections = sv.Detections.from_inference(res_w)
        
        if len(detections) > 0:
            detections = tracker.update_with_detections(detections)
            
            # Verificacion de seguridad para evitar errores NoneType
            if detections.tracker_id is not None:
                labels = [f"ARMA {c:.2f}" for c in detections.confidence]
                frame = box_annotator.annotate(scene=frame, detections=detections)
                frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
                
                # Alerta Armas
                if (current_time - last_w_db) > COOLDOWN_DB:
                    url = ""
                    if (current_time - last_w_img) > COOLDOWN_IMG:
                        print("!!! ARMA DETECTADA - FOTO !!!")
                        url = subir_a_imgbb(frame, "arma")
                        last_w_img = current_time
                    insert_weapon_db(detections.tracker_id[0], detections['class_name'][0], url)
                    last_w_db = current_time
    except Exception: pass

    try:
        # Confianza baja (0.3) para captar personas desde arriba
        results_pose = model_pose.predict(frame, conf=0.3, verbose=False)[0]
        keypoints_all = []
        
        if results_pose.boxes:
            for i, kp in enumerate(results_pose.keypoints):
                pts = kp.data[0].cpu().numpy()
                if len(pts) > 0:
                    kps_list = [(int(x), int(y)) for x, y, c in pts]
                    keypoints_all.append(kps_list)
                    
                    frame = draw_skeleton(frame, kps_list)
                    
                    center = get_centroid(kps_list)
                    pid = (int(center[0]//50), int(center[1]//50))
                    
                    hand_avg = (np.array(kps_list[9]) + np.array(kps_list[10])) / 2
                    hand_histories[pid].append(hand_avg)

            agresion_detectada = False
            tipo_agresion = ""

            for i in range(len(keypoints_all)):
                for j in range(i + 1, len(keypoints_all)):
                    c1 = get_centroid(keypoints_all[i])
                    c2 = get_centroid(keypoints_all[j])
                    
                    pid1 = (int(c1[0]//50), int(c1[1]//50))
                    pid2 = (int(c2[0]//50), int(c2[1]//50))
                    s1 = hand_speed(hand_histories[pid1])
                    s2 = hand_speed(hand_histories[pid2])
                    
                    is_agresion, tipo = detect_aggression_cctv(
                        keypoints_all[i], keypoints_all[j], s1, s2
                    )
                    
                    if is_agresion:
                        agresion_detectada = True
                        tipo_agresion = tipo
                        cv2.line(frame, tuple(c1), tuple(c2), (0, 0, 255), 4)
                        cv2.putText(frame, f"ALERTA: {tipo}", (50, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

            if agresion_detectada:
                 if (current_time - last_a_db) > COOLDOWN_DB:
                    url = ""
                    if (current_time - last_a_img) > COOLDOWN_IMG:
                        print(f"!!! {tipo_agresion} - FOTO !!!")
                        url = subir_a_imgbb(frame, "agresion")
                        last_a_img = current_time
                    insert_aggression_db(frame_count, len(keypoints_all), tipo_agresion, url)
                    last_a_db = current_time

    except Exception as e: pass

    out.write(frame)
    
    if frame_count % 10 == 0:
        clear_output(wait=True)
        print(f"Procesando Frame: {frame_count}")
        cv2_imshow(cv2.resize(frame, (480, 270)))

cap.release()
out.release()
print(f"Finalizado. Descarga: {OUTPUT_NAME}")