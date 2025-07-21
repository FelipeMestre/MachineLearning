import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import os

# Configuración de visualización
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

def load_yolo():
    """Carga el modelo YOLO pre-entrenado"""
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i-1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

def detect_objects(img, net, output_layers):
    """Detecta objetos en la imagen usando YOLO"""
    height, width, channels = img.shape
    
    # Preprocesamiento de la imagen
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Procesamiento de las detecciones
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Umbral de confianza
                # Coordenadas del objeto
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Coordenadas del rectángulo
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Aplicar Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, confidences, class_ids, indexes

def draw_labels(img, boxes, confidences, class_ids, indexes, classes):
    """Dibuja las cajas delimitadoras y etiquetas en la imagen"""
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), font, 1, color, 2)
    
    return img

def main():
    # Cargar YOLO
    print("Cargando YOLO...")
    net, classes, output_layers = load_yolo()
    
    # Cargar imagen
    image_path = "example.jpg"
    if not os.path.exists(image_path):
        print("Por favor, proporciona una imagen para probar el detector")
        return
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detectar objetos
    print("Detectando objetos...")
    start_time = time.time()
    boxes, confidences, class_ids, indexes = detect_objects(img, net, output_layers)
    end_time = time.time()
    
    print(f"Tiempo de detección: {end_time - start_time:.2f} segundos")
    
    # Dibujar resultados
    img = draw_labels(img, boxes, confidences, class_ids, indexes, classes)
    
    # Mostrar resultados
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main() 