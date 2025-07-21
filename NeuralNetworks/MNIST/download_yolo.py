import urllib.request
import os
import ssl

# Deshabilitar verificación SSL para evitar problemas de certificado
ssl._create_default_https_context = ssl._create_unverified_context

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Descargando {filename}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print("¡Descarga completada!")
        except Exception as e:
            print(f"Error al descargar {filename}: {str(e)}")
    else:
        print(f"{filename} ya existe.")

# URLs de los archivos de YOLOv3
yolo_files = {
    "yolov3.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights",
    "yolov3.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3.cfg",
    "coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/coco.names"
}

# Descargar archivos
for filename, url in yolo_files.items():
    download_file(url, filename)

print("¡Todos los archivos necesarios han sido descargados!") 