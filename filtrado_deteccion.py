# Con este podés probar el procesamiento para las imágenes
# filtra, detecta rois y números

import cv2
import numpy as np
import matplotlib.pyplot as plt

def procesar_roi_numeros(roi, mascara_roi, x, y, w, h, copia_original, rois_numeros_filtrados, rois_clausuras):
    """
    Procesa una ROI para detectar y contar números.

    Args:
        frame: Frame original
        roi: Región de interés extraída
        mascara_roi: Máscara de números recortada para la ROI
        x, y, w, h: Coordenadas del bounding box
        copia_original: Imagen donde dibujar los resultados
        rois_numeros_filtrados: Lista para almacenar ROIs filtradas
        rois_clausuras: Lista para almacenar ROIs con clausura aplicada

    Returns:
        num_contornos: Número de contornos detectados
    """
    # Aplicar filtrado de números a la ROI
    roi_numero_filtrado = cv2.bitwise_and(roi, roi, mask=mascara_roi)

    # Convertir a escala de grises para operaciones morfológicas
    roi_gray = cv2.cvtColor(roi_numero_filtrado, cv2.COLOR_BGR2GRAY)

    # Binarizar
    _, roi_binaria = cv2.threshold(roi_gray, 1, 255, cv2.THRESH_BINARY)

    # Aplicar clausura morfológica
    s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    roi_clausura = cv2.morphologyEx(roi_binaria, cv2.MORPH_CLOSE, s, iterations=3)

    # Encontrar contornos internos (ahora sobre imagen binaria)
    contornos_numeros, jerarquia = cv2.findContours(roi_clausura, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Contar los contornos detectados
    num_contornos = len(contornos_numeros)

    # Dibujar en la imagen con el número de contornos
    cv2.rectangle(copia_original, (x, y), (x+w, y+h), (128, 0, 128), 2)
    cv2.putText(copia_original, str(num_contornos), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2)

    rois_numeros_filtrados.append(roi_numero_filtrado)
    rois_clausuras.append(roi_clausura)

    return num_contornos

win_frame = 'Frame'
win_mask = 'Mascara'
win_frame_filtrado = 'Frame filtrado'
win_frame_filtrado_neg = 'Frame filtrado negativo'

H_low = 58
S_low = 90
V_low = 0
H_high = 150
S_high = 255
V_high = 202

H_low_numero = 0
H_high_numero = 180
S_low_numero = 0
S_high_numero = 113
V_low_numero = 200
V_high_numero = 255

frame = cv2.imread('C:\\Users\\PC\\Desktop\\Procesamiento de Imágenes I\\TPs\\TP3\\test.jpg')

# --- Proceso -----------------------------------------------------
frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
frame_threshold = cv2.inRange(frame_HSV, (H_low, S_low, V_low), (H_high, S_high, V_high))
frame_threshold_numero = cv2.inRange(frame_HSV, (H_low_numero, S_low_numero, V_low_numero), (H_high_numero, S_high_numero, V_high_numero))
frame_filtrado = cv2.bitwise_and(frame, frame, mask=frame_threshold)
frame_filtrado_neg = cv2.bitwise_and(frame, frame, mask=~frame_threshold)
frame_filtrado_neg_gray = cv2.cvtColor(frame_filtrado_neg, cv2.COLOR_BGR2GRAY)

kernel = np.ones((5, 5), np.uint8)
mascara = cv2.morphologyEx(frame_filtrado_neg_gray, cv2.MORPH_CLOSE, kernel)  # Cerrar huecos
_, mascara_binaria = cv2.threshold(mascara, 0, 255, cv2.THRESH_BINARY)

contornos, _ = cv2.findContours(mascara_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rois = []
rois_numeros_filtrados = []
rois_clausuras = []
centroides = []

copia_original = frame.copy()
for i, contorno in enumerate(contornos):
    x, y, w, h = cv2.boundingRect(contorno)

    # Filtrar contornos muy pequeños
    if w * h < 100:
        continue

    # Calcular el centroide usando el centro del bounding box
    cx = x + w // 2
    cy = y + h // 2

    centroides.append((cx, cy))

    # A partir de acá, hacerlo sólo si los centroides no coinciden (o un umbral)

    # Extraer ROI
    roi = frame[y:y+h, x:x+w]
    rois.append(roi)

    # Recortar la máscara de números para que coincida con la ROI
    mascara_roi = frame_threshold_numero[y:y+h, x:x+w]

    # Aplicar filtrado de números a la ROI
    roi_numero_filtrado = cv2.bitwise_and(roi, roi, mask=mascara_roi)

    # Convertir a escala de grises para operaciones morfológicas
    roi_gray = cv2.cvtColor(roi_numero_filtrado, cv2.COLOR_BGR2GRAY)

    # Binarizar
    _, roi_binaria = cv2.threshold(roi_gray, 1, 255, cv2.THRESH_BINARY)

    # Aplicar apertura y clausura morfológica
    sa = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    roi_apertura = cv2.morphologyEx(mascara_roi, cv2.MORPH_OPEN, sa, iterations=1)
    roi_clausura = cv2.morphologyEx(roi_apertura, cv2.MORPH_CLOSE, s, iterations=2)

    # Encontrar contornos internos (ahora sobre imagen binaria)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_clausura, 8, cv2.CV_32S)
    num_contornos = len(centroids)-1

    # Dibujar en la imagen con el número de contornos
    cv2.rectangle(copia_original, (x, y), (x+w, y+h), (128, 0, 128), 2)
    cv2.putText(copia_original, str(num_contornos), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2)

    rois_numeros_filtrados.append(roi_numero_filtrado)
    rois_clausuras.append(roi_clausura)

# Visualizar con matplotlib
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# Fila 1
axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Frame Original')
axes[0, 0].axis('off')

axes[0, 1].imshow(frame_threshold, cmap='gray')
axes[0, 1].set_title('Máscara')
axes[0, 1].axis('off')

# Fila 2
axes[1, 0].imshow(cv2.cvtColor(frame_filtrado, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('Frame Filtrado')
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(frame_filtrado_neg, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('Frame Filtrado Negativo')
axes[1, 1].axis('off')

# Fila 3
axes[2, 0].imshow(mascara_binaria, cmap='gray')
axes[2, 0].set_title('Máscara Binaria')
axes[2, 0].axis('off')

axes[2, 1].imshow(cv2.cvtColor(copia_original, cv2.COLOR_BGR2RGB))
axes[2, 1].set_title('ROIs Detectadas')
axes[2, 1].axis('off')

plt.tight_layout()
plt.show()

# Visualizar todas las ROIs con filtrado de números
if len(rois_numeros_filtrados) > 0:
    # Calcular el número de filas necesarias (3 ROIs por fila)
    num_rois = len(rois_numeros_filtrados)
    cols = 3
    rows = (num_rois + cols - 1) // cols  # Redondear hacia arriba

    fig2, axes2 = plt.subplots(rows, cols, figsize=(15, 5*rows))

    # Aplanar el array de axes para facilitar la iteración
    if rows == 1:
        axes2 = [axes2] if cols == 1 else axes2
    else:
        axes2 = axes2.flatten()

    for i, roi_filtrado in enumerate(rois_numeros_filtrados):
        axes2[i].imshow(cv2.cvtColor(roi_filtrado, cv2.COLOR_BGR2RGB))
        axes2[i].set_title(f'ROI {i} - Números Filtrados')
        axes2[i].axis('off')

    # Ocultar subplots vacíos
    for j in range(num_rois, len(axes2)):
        axes2[j].axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("No se detectaron ROIs para mostrar.")

# Visualizar todas las ROIs con clausura aplicada
if len(rois_clausuras) > 0:
    # Calcular el número de filas necesarias (3 ROIs por fila)
    num_rois = len(rois_clausuras)
    cols = 3
    rows = (num_rois + cols - 1) // cols  # Redondear hacia arriba

    fig3, axes3 = plt.subplots(rows, cols, figsize=(15, 5*rows))

    # Aplanar el array de axes para facilitar la iteración
    if rows == 1:
        axes3 = [axes3] if cols == 1 else axes3
    else:
        axes3 = axes3.flatten()

    for i, roi_clausura in enumerate(rois_clausuras):
        axes3[i].imshow(roi_clausura, cmap='gray')
        axes3[i].set_title(f'ROI {i} - Clausura')
        axes3[i].axis('off')

    # Ocultar subplots vacíos
    for j in range(num_rois, len(axes3)):
        axes3[j].axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("No se detectaron ROIs con clausura para mostrar.")
