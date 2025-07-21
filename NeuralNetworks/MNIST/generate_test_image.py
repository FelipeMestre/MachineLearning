import cv2
import numpy as np

# Crear una imagen en blanco
image = np.ones((500, 800, 3), dtype=np.uint8) * 255  # Fondo blanco

# Dibujar una persona (silueta)
cv2.rectangle(image, (100, 200), (150, 400), (0, 0, 0), -1)  # Cuerpo
cv2.circle(image, (125, 150), 25, (0, 0, 0), -1)  # Cabeza

# Dibujar un coche
cv2.rectangle(image, (300, 300), (500, 350), (255, 0, 0), -1)  # Cuerpo del coche
cv2.circle(image, (350, 350), 20, (0, 0, 0), -1)  # Rueda 1
cv2.circle(image, (450, 350), 20, (0, 0, 0), -1)  # Rueda 2

# Dibujar una bicicleta
cv2.circle(image, (600, 300), 30, (0, 255, 0), -1)  # Rueda trasera
cv2.circle(image, (700, 300), 30, (0, 255, 0), -1)  # Rueda delantera
cv2.line(image, (600, 300), (700, 300), (0, 255, 0), 3)  # Barra horizontal
cv2.line(image, (650, 200), (700, 300), (0, 255, 0), 3)  # Manillar

# Guardar la imagen
cv2.imwrite('example.jpg', image)

print("Imagen de prueba generada como 'example.jpg'") 