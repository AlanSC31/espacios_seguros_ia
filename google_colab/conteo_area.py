from ultralytics import YOLO
import cv2
import numpy as np
import time
import mysql.connector
from collections import defaultdict

# Conexión a la base de datos
db = mysql.connector.connect(
    host="sql5.freesqldatabase.com",
    user="sql5782791",
    password="rWhInNN5gt",
    database="sql5782791"
)
cursor = db.cursor()

# Modelo YOLO
model = YOLO("yolov8m.pt")

# source
cap = cv2.VideoCapture("personas_calle_3.mp4")

# Configuración de output
width, height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('conteo_area.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# params
grid_rows, grid_cols = 2, 2
cell_height, cell_width = height // grid_rows, width // grid_cols
umbral_concurrido = 5
intervalo_zona = 5
intervalo_conteo = 3
intervalo_batch = 5  #  insertar a BD cada 5 seg
# states
tiempo_celda = defaultdict(lambda: defaultdict(float))
celda_actual = {}
personas_detectadas = set()
ultima_insercion_batch = time.time()

# Buffers de datos para batch insert
buffer_zona = []
buffer_conteo = []
buffer_alertas = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    tiempo_actual = time.time()
    grid_counts = np.zeros((grid_rows, grid_cols), dtype=int)

    results = model.track(frame, persist=True, conf=0.4, classes=[0], tracker="bytetrack.yaml", verbose=False)

    if results and results[0].boxes.id is not None:
        ids = results[0].boxes.id.int().tolist()
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for i, (box, id_persona) in enumerate(zip(boxes, ids)):
            id_persona = int(id_persona)
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            fila, col = cy // cell_height, cx // cell_width

            if fila >= grid_rows or col >= grid_cols:
                continue

            grid_counts[fila, col] += 1

            if id_persona not in personas_detectadas:
                cursor.execute("INSERT INTO personas_detectadas (persona_id, timestamp_detectado) VALUES (%s, NOW())", (id_persona,))
                personas_detectadas.add(id_persona)

            celda_prev = celda_actual.get(id_persona)
            if celda_prev == (fila, col):
                tiempo_celda[id_persona][(fila, col)] += 1 / fps
            else:
                celda_actual[id_persona] = (fila, col)
                tiempo_celda[id_persona][(fila, col)] = 0

            tiempo_en_zona = tiempo_celda[id_persona][(fila, col)]

            # ⌛ Guardar en buffer (NO en la base de datos todavía)
            if int(tiempo_en_zona) % intervalo_zona == 0:
                buffer_zona.append((id_persona, fila, col, int(tiempo_en_zona)))

            cv2.putText(frame, f"ID:{id_persona} T:{int(tiempo_en_zona)}s", (cx, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Conteo y alertas por celda
    for i in range(grid_rows):
        for j in range(grid_cols):
            cantidad = int(grid_counts[i, j])
            if cantidad > 0 and int(tiempo_actual) % intervalo_conteo == 0:
                buffer_conteo.append((i, j, cantidad))
                if cantidad >= umbral_concurrido:
                    buffer_alertas.append((i, j, cantidad))

            x1, y1 = j * cell_width, i * cell_height
            x2, y2 = x1 + cell_width, y1 + cell_height
            color = (0, 0, 255) if cantidad >= umbral_concurrido else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cantidad} p", (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # INSERTAR A BASE DE DATOS CADA N SEGUNDOS
    if tiempo_actual - ultima_insercion_batch >= intervalo_batch:
        if buffer_zona:
            cursor.executemany(
                "INSERT INTO tiempo_en_zona (persona_id, fila, columna, tiempo_segundos, timestamp) VALUES (%s, %s, %s, %s, NOW())",
                buffer_zona
            )
            buffer_zona.clear()

        if buffer_conteo:
            cursor.executemany(
                "INSERT INTO conteo_zona (fila, columna, cantidad, timestamp) VALUES (%s, %s, %s, NOW())",
                buffer_conteo
            )
            buffer_conteo.clear()

        if buffer_alertas:
            cursor.executemany(
                "INSERT INTO alertas_congestion (fila, columna, cantidad, fecha_creacion) VALUES (%s, %s, %s, NOW())",
                buffer_alertas
            )
            buffer_alertas.clear()

        db.commit()
        ultima_insercion_batch = tiempo_actual

    out.write(frame)

# Liberar recursos
cap.release()
out.release()
cursor.close()
db.close()
