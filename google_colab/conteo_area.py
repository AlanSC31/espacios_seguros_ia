from ultralytics import YOLO
import cv2
import numpy as np
import time
import mysql.connector
from collections import defaultdict
import sys 

def conectar_bd():
    try:
        db = mysql.connector.connect(
            host="2806:2f0:54c1:f424:7254:d2ff:fe3c:6399",
            user="dries",
            password="=rqr:WkooWsKTZC@:0Bb@@rryaD)H+7m08AbU!Hf!}]2,8%XiNz*Rpp]0Q2B9eaJjb4u#=uY5,)f~4TJhQ%sHD]v>q}dP~g.rryz",
            database="esia"
        )
        print("conexcion db exitosa")
        return db, db.cursor()
    except mysql.connector.Error as err:
        print(f"error al conectar a la db: {err}")
        sys.exit(1) 

def main():
    
    MODEL_PATH = "yolov8m.pt"
    INPUT_VIDEO_PATH = "personas_calle_3.mp4" # "0" para webcam
    OUTPUT_VIDEO_PATH = "conteo_area_output.mp4"

    db = None
    cursor = None
    cap = None
    out = None

    try:
        db, cursor = conectar_bd()

        # modelo YOLO 
        print(f"cargando modelo {MODEL_PATH}...")
        model = YOLO(MODEL_PATH)

        # source video 
        source = 0 if INPUT_VIDEO_PATH == "0" else INPUT_VIDEO_PATH
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"No se pudo abrir la fuente de video: {INPUT_VIDEO_PATH}")

        # configuracion del video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) # FPS para el video de salida
        
        print(f"Video de entrada: {width}x{height} @ {fps_video:.2f} FPS")

        # video output 
        out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps_video, (width, height))

        grid_rows, grid_cols = 2, 2
        cell_height, cell_width = height // grid_rows, width // grid_cols
        umbral_concurrido = 5
        intervalo_zona = 5     # Loggear tiempo en zona cada 5s
        intervalo_conteo = 3   # Loggear conteo de zona cada 3s
        intervalo_batch = 5    # Insertar a BD cada 5 seg
        
        tiempo_celda = defaultdict(lambda: defaultdict(float))
        tiempo_log_zona = defaultdict(lambda: defaultdict(int)) # Almacena el ultimo umbral loggeado (5s, 10s, etc)
        celda_actual = {}
        personas_detectadas = set() # Cache de IDs ya insertados
        
        # Buffers de datos para batch insert
        buffer_zona = []
        buffer_conteo = []
        buffer_alertas = []

        # Timers para logica de intervalos
        ultima_insercion_batch = time.time()
        ultima_insercion_conteo = time.time()
        tiempo_anterior_frame = time.time() # Para calculo de delta_time

        print("Iniciando procesamiento...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Fin del video.")
                break

            # Calculo de Tiempo Preciso 
            tiempo_actual = time.time()
            delta_time = tiempo_actual - tiempo_anterior_frame
            tiempo_anterior_frame = tiempo_actual

            # Deteccion y Seguimiento
            grid_counts = np.zeros((grid_rows, grid_cols), dtype=int)
            results = model.track(frame, persist=True, conf=0.4, classes=[0], tracker="bytetrack.yaml", verbose=False)

            if results and results[0].boxes.id is not None:
                ids = results[0].boxes.id.int().tolist()
                boxes = results[0].boxes.xyxy.cpu().numpy()

                for i, (box, id_persona) in enumerate(zip(boxes, ids)):
                    id_persona = int(id_persona)
                    x1, y1, x2, y2 = map(int, box)

                    # Precision de Zona: Usar centro inferior
                    cx = (x1 + x2) // 2
                    cy = y2 # Usar el punto mas bajo del buzon

                    fila, col = cy // cell_height, cx // cell_width

                    if fila >= grid_rows or col >= grid_cols:
                        continue

                    grid_counts[fila, col] += 1

                    # Registrar persona nueva si no esta en cache
                    if id_persona not in personas_detectadas:
                        try:
                            cursor.execute("INSERT INTO personas_detectadas (persona_id, timestamp_detectado) VALUES (%s, NOW())", (id_persona,))
                            personas_detectadas.add(id_persona)
                        except mysql.connector.Error as err:
                            # Ignorar errores de duplicados si el script se reinicia
                            if err.errno == 1062: # Duplicate entry
                                personas_detectadas.add(id_persona)
                            else:
                                print(f"Error en BD (personas_detectadas): {err}")

                    # Precision de Tiempo Usar delta_time 
                    if celda_actual.get(id_persona) != (fila, col):
                        # Si la persona cambio de celda, resetear su tiempo base
                        celda_actual[id_persona] = (fila, col)
                        tiempo_celda[id_persona][(fila, col)] = 0
                        tiempo_log_zona[id_persona][(fila, col)] = 0 # Resetear log
                    else:
                        # Si sigue en la misma celda, acumular tiempo real
                        tiempo_celda[id_persona][(fila, col)] += delta_time

                    tiempo_en_zona = tiempo_celda[id_persona][(fila, col)]

                    ultimo_log = tiempo_log_zona[id_persona][(fila, col)]
                    if int(tiempo_en_zona) >= ultimo_log + intervalo_zona:
                        nuevo_log_time = ultimo_log + intervalo_zona
                        buffer_zona.append((id_persona, fila, col, nuevo_log_time))
                        tiempo_log_zona[id_persona][(fila, col)] = nuevo_log_time

                    cv2.putText(frame, f"ID:{id_persona} T:{int(tiempo_en_zona)}s", (cx, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Conteo y Alertas por Celda 
            enviar_conteo = False
            if tiempo_actual - ultima_insercion_conteo >= intervalo_conteo:
                enviar_conteo = True
                ultima_insercion_conteo = tiempo_actual

            for i in range(grid_rows):
                for j in range(grid_cols):
                    cantidad = int(grid_counts[i, j])
                    
                    if cantidad > 0 and enviar_conteo:
                        buffer_conteo.append((i, j, cantidad))
                        if cantidad >= umbral_concurrido:
                            buffer_alertas.append((i, j, cantidad))

                    # Dibujo de la cuadricula
                    x1, y1 = j * cell_width, i * cell_height
                    x2, y2 = x1 + cell_width, y1 + cell_height
                    color = (0, 0, 255) if cantidad >= umbral_concurrido else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{cantidad} p", (x1 + 5, y1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # inserts en db por lotes 
            if tiempo_actual - ultima_insercion_batch >= intervalo_batch:
                try:
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

                    if any([buffer_zona, buffer_conteo, buffer_alertas]):
                        db.commit()
                        
                except mysql.connector.Error as err:
                    print(f"Error al insertar en BD: {err}")
                    db.rollback() # Revertir si hay un error en el lote

                ultima_insercion_batch = tiempo_actual

            # Escribir frame de salida
            out.write(frame)
            
            # Opcional: Mostrar video en vivo
            # cv2.imshow("Frame", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    except Exception as e:
        print(f"Ocurrio un error inesperado: {e}")
    
    finally:
        # liberar recursos
        print("\nLiberando recursos...")
        if cap:
            cap.release()
            print("Fuente de video liberada.")
        if out:
            out.release()
            print("Archivo de salida cerrado.")
        if cursor:
            cursor.close()
            print("Cursor de BD cerrado.")
        if db:
            # Asegurarse de hacer commit de lo ultimo antes de cerrar
            try:
                if any([buffer_zona, buffer_conteo, buffer_alertas]):
                    print("Guardando buffers finales en BD...")
                    # Repetir la logica de insercion una ultima vez
                    if buffer_zona:
                        cursor.executemany("INSERT INTO tiempo_en_zona (persona_id, fila, columna, tiempo_segundos, timestamp) VALUES (%s, %s, %s, %s, NOW())", buffer_zona)
                    if buffer_conteo:
                        cursor.executemany("INSERT INTO conteo_zona (fila, columna, cantidad, timestamp) VALUES (%s, %s, %s, NOW())", buffer_conteo)
                    if buffer_alertas:
                        cursor.executemany("INSERT INTO alertas_congestion (fila, columna, cantidad, fecha_creacion) VALUES (%s, %s, %s, NOW())", buffer_alertas)
                    db.commit()
                    print("Buffers finales guardados.")
            except mysql.connector.Error as err:
                print(f"Error al guardar buffers finales: {err}")
                db.rollback()
            
            db.close()
            print("Conexion de BD cerrada.")
        
        cv2.destroyAllWindows()
        print("Proceso completado.")

if __name__ == "__main__":
    main()