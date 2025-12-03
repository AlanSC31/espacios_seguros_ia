from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict, deque
import mysql.connector
from datetime import datetime

# conexion a la base de datos
db = mysql.connector.connect(
    host="2806:2f0:54c1:f424:7254:d2ff:fe3c:6399",
    user="dries",
    password="mosca_311",
    database="esia"
)
cursor = db.cursor()

input_video = 'pelea_2.mp4'
output_video = f"procesado_{input_video}"

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def hand_speed(history):
    if len(history) < 2:
        return 0
    speeds = [euclidean(history[i], history[i+1]) for i in range(len(history)-1)]
    return np.mean(speeds)

def torso_speed(history):
    if len(history) < 2:
        return 0
    speeds = [euclidean(history[i], history[i+1]) for i in range(len(history)-1)]
    return np.mean(speeds)

def detect_aggression(pose1, pose2, hand_speed1, hand_speed2, torso_hist1, torso_hist2):
    d = euclidean(pose1[2], pose2[2])
    if d > 30:
        return False

    head2 = pose2[0]
    hand_r = pose1[9]
    hand_l = pose1[10]
    golpe_cabeza = euclidean(hand_r, head2) < 60 or euclidean(hand_l, head2) < 60

    movimiento_agresivo = hand_speed1 > 10 and hand_speed2 > 5

    velocidad_torso1 = torso_speed(torso_hist1)
    velocidad_torso2 = torso_speed(torso_hist2)
    empujon = velocidad_torso1 > 10 or velocidad_torso2 > 10

    jaloneo = golpe_cabeza and (hand_speed1 > 5 or hand_speed2 > 5)

    torso_movimiento_suficiente = velocidad_torso1 > 3 or velocidad_torso2 > 3

    return golpe_cabeza or (movimiento_agresivo and torso_movimiento_suficiente and d < 30) or empujon or jaloneo

model = YOLO("yolo11x-pose.pt")

cap = cv2.VideoCapture(input_video)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

hand_histories = defaultdict(lambda: deque(maxlen=8))
torso_histories = defaultdict(lambda: deque(maxlen=8))

print("procesando video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.4, verbose=False)[0]
    keypoints_all = []
    boxes = []

    for i, kp in enumerate(results.keypoints):
        pts = kp.data[0].cpu().numpy()
        person_pose = [(int(x), int(y)) for x, y, c in pts]
        keypoints_all.append(person_pose)

        box = results.boxes[i].xyxy[0].cpu().numpy().astype(int)
        boxes.append(box)

        person_id = tuple(box)
        hand_histories[person_id].append(person_pose[9])
        torso_histories[person_id].append(person_pose[2])

    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    agresion_detectada_en_frame = False

    for i in range(len(keypoints_all)):
        for j in range(i + 1, len(keypoints_all)):
            id_i = tuple(boxes[i])
            id_j = tuple(boxes[j])

            hs_i = hand_speed(hand_histories[id_i])
            hs_j = hand_speed(hand_histories[id_j])
            th_i = torso_histories[id_i]
            th_j = torso_histories[id_j]

            # detccion de agresion
            if detect_aggression(keypoints_all[i], keypoints_all[j], hs_i, hs_j, th_i, th_j):
                agresion_detectada_en_frame = True
                x1, y1, x2, y2 = boxes[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                x1, y1, x2, y2 = boxes[j]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "agresion detectada", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)

                # definir tipos de agresion 
                d = euclidean(keypoints_all[i][2], keypoints_all[j][2])
                head2 = keypoints_all[j][0]
                hand_r = keypoints_all[i][9]
                hand_l = keypoints_all[i][10]
                golpe_cabeza = euclidean(hand_r, head2) < 60 or euclidean(hand_l, head2) < 60
                movimiento_agresivo = hs_i > 10 and hs_j > 5
                velocidad_torso1 = torso_speed(th_i)
                velocidad_torso2 = torso_speed(th_j)
                empujon = velocidad_torso1 > 10 or velocidad_torso2 > 10
                jaloneo = golpe_cabeza and (hs_i > 5 or hs_j > 5)

                tipos = []
                if golpe_cabeza: tipos.append("golpe_cabeza")
                if movimiento_agresivo: tipos.append("movimiento_agresivo")
                if empujon: tipos.append("empujon")
                if jaloneo: tipos.append("jaloneo")

                involucrados_str = f"{id_i},{id_j}"
                tipo_agresion_str = ",".join(tipos)
                velocidad_manos_str = f"{hs_i},{hs_j}"
                velocidad_torsos_str = f"{velocidad_torso1},{velocidad_torso2}"

                # insertar en BD
                sql = """
                INSERT INTO agresiones_detectadas 
                (timestamp, video_nombre, frame_num, cant_personas, id_involucrados, tipo_agresion, distancia_entre, velocidad_manos, velocidad_torsos)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                data = (
                    datetime.now(),
                    input_video,
                    frame_number,
                    len(keypoints_all),
                    involucrados_str,
                    tipo_agresion_str,
                    d,
                    velocidad_manos_str,
                    velocidad_torsos_str
                )
                cursor.execute(sql, data)
                db.commit()

    out.write(frame)

cap.release()
out.release()
cursor.close()
db.close()
print("video procesado guardado:", output_video)
