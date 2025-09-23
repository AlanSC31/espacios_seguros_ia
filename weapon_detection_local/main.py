from roboflow import Roboflow
import os
import cv2

# Conexión con Roboflow
rf = Roboflow(api_key="5b3nLnlY9wMoNfAcQ51R")
project = rf.workspace("m-qczea").project("weapon_detection_v2-rdpq3")
version = project.version(2)
model = version.model

# Cargar video de entrada
cap = cv2.VideoCapture("./src/video3.mp4")

# Definir resolución compatible con el modelo (640x640)
output_size = (640, 640)
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("detectado2.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)

# Procesar cada frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, output_size)
    cv2.imwrite("temp.jpg", resized)

    model.predict("temp.jpg", confidence=30, overlap=30).save("temp_out.jpg")

    if not os.path.exists("temp_out.jpg"):
        print(" No se generó temp_out.jpg")
        continue

    frame_out = cv2.imread("temp_out.jpg")
    if frame_out is None:
        print(" No se pudo leer temp_out.jpg")
        continue

    out.write(frame_out)


    # cv2.imshow("Detección", frame_out)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
out.release()
# cv2.destroyAllWindows()  # Solo si usas imshow
