# Monitoreo espacios publicos

## Descripcion
Este proyecto implementa un sistema de vision por computadora utilizando YOLO y DeepSORT para el monitoreo inteligente y automatizado de espacios publicos. El objetivo principal es identificar situaciones de riesgo y anomalias en tiempo real a traves de camaras de seguridad.

El modelo esta entrenado/configurado para detectar:
* Personas
* Situaciones de agresion
* Presencia de armas
* Acumulacion de basura

Ademas de la deteccion, el sistema esta preparado para integrarse con una aplicacion movil, lo que permite enviar notificaciones push automaticas a los usuarios o autoridades correspondientes en el momento en que se registra un incidente.

## Get started

Los scripts se pueden ejecutar de dos maneras: de forma local, viendo en tiempo real el procesamiento del *source* (lo cual consume muchos recursos), y en la nube, utilizando *Google Colab*.

### Local

#### Requisitos:
1. Descargar el modelo YOLO (por ejemplo, la version 8m): [Aqui](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt) y colocarlo en una carpeta llamada `models` en la raiz del proyecto. *(Nota: Asegurate de que el nombre del modelo coincida con el que llamas en tu script)*.
2. Python 3.13.3 o superior.
3. Librerias de Python: `ultralytics` y `deep_sort_realtime`.

#### Instalacion y ejecucion:
1. Instalar Python.
2. Instalar las dependencias necesarias ejecutando en tu terminal:

    pip install ultralytics deep_sort_realtime

3. Navegar en la terminal hasta la carpeta raiz del proyecto y ejecutar el script principal:

    python nombre_del_archivo.py

### Google Colab

#### Instalacion y ejecucion:
1. Abrir una sesion de Google Colab.
2. Descargar el modelo YOLOv8x: [Aqui](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt)
3. Subir los archivos del proyecto y el modelo `.pt` a la raiz del entorno en Google Colab.
4. Instalar las librerias necesarias en una celda de codigo:

    pip install ultralytics deep_sort_realtime

5. Ejecutar el script principal con:

    !python nombre_del_archivo.py

Esto generara el *output* directamente en la carpeta raiz del entorno.