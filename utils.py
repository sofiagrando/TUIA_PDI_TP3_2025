# Este tiene las funciones necesarias para procesar el video

import cv2
import numpy as np
import math
from collections import Counter

win_frame = 'Frame'
win_mask = 'Mascara'
win_frame_filtrado = 'Frame filtrado'
win_frame_filtrado_neg = 'Frame filtrado negativo'

H_low_verde = 30
H_high_verde = 90

S_low_verde = 140
S_high_verde = 255

V_low_verde = 0
V_high_verde = 255

H1_low = 0
H1_high = 30

H2_low = 160
H2_high = 180

S_low = 75
V_low = 75
S_high = 255
V_high = 255

H_low_numero = 0
H_high_numero = 180
S_low_numero = 0
S_high_numero = 110
V_low_numero = 180
V_high_numero = 255

# Filtros de área para dados rojos
area_min_dado = 500
area_max_dado = 1000

def procesar_frame(frame):
    """
    Procesa un frame completo para detectar dados rojos dentro de contenedores verdes.
    Pipeline:
        1. Detecta contenedores verdes
        2. Dentro de cada contenedor, detecta dados rojos
        3. Retorna las coordenadas de los dados rojos (en coordenadas absolutas del frame)

    Args:
        frame: Frame BGR a procesar

    Returns:
        contornos_dados_rojos: Lista de contornos de dados rojos detectados
        frame_threshold_numero: Máscara de números blancos del frame (para usar en procesar_roi_numeros)
        centroides: Lista de tuplas (cx, cy) con los centroides de cada dado rojo
        boxes: Lista de tuplas (x_abs, y_abs, w, h) con las bounding boxes de dados rojos en coordenadas absolutas
    """
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Crear máscara de números blancos (se mantiene del código original)
    frame_threshold_numero = cv2.inRange(frame_HSV, (H_low_numero, S_low_numero, V_low_numero), (H_high_numero, S_high_numero, V_high_numero))

    # PASO 1: Filtrar por VERDE para encontrar contenedores (dados)
    mascara_verde = cv2.inRange(frame_HSV,
                                (H_low_verde, S_low_verde, V_low_verde),
                                (H_high_verde, S_high_verde, V_high_verde))

    # Aplicar operaciones morfológicas para limpiar la máscara verde
    kernel = np.ones((5, 5), np.uint8)
    mascara_verde_limpia = cv2.morphologyEx(mascara_verde, cv2.MORPH_CLOSE, kernel)

    # Binarizar
    _, mascara_verde_binaria = cv2.threshold(mascara_verde_limpia, 0, 255, cv2.THRESH_BINARY)

    # PASO 2: Encontrar contornos de las ROIs verdes
    contornos_verdes, _ = cv2.findContours(mascara_verde_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # PASO 3: Para cada ROI verde (contenedor), detectar dados rojos
    centroides = []
    boxes = []
    contornos_dados_rojos = []

    for contorno in contornos_verdes:
        x_verde, y_verde, w_verde, h_verde = cv2.boundingRect(contorno)

        # Extraer ROI verde del frame original (mantiene todos los colores)
        roi_contenedor = frame[y_verde:y_verde+h_verde, x_verde:x_verde+w_verde]
        roi_contenedor_HSV = cv2.cvtColor(roi_contenedor, cv2.COLOR_BGR2HSV)

        # FILTRAR POR ROJO dentro del contenedor verde (doble rango para rojo)
        mascara_rojo_1 = cv2.inRange(roi_contenedor_HSV, (H1_low, S_low, V_low), (H1_high, S_high, V_high))
        mascara_rojo_2 = cv2.inRange(roi_contenedor_HSV, (H2_low, S_low, V_low), (H2_high, S_high, V_high))
        mascara_rojo = cv2.bitwise_or(mascara_rojo_1, mascara_rojo_2)

        # Aplicar operaciones morfológicas para limpiar la máscara roja
        mascara_rojo_limpia = cv2.morphologyEx(mascara_rojo, cv2.MORPH_CLOSE, kernel)
        _, mascara_rojo_binaria = cv2.threshold(mascara_rojo_limpia, 0, 255, cv2.THRESH_BINARY)

        # Encontrar contornos de dados rojos dentro del contenedor verde
        contornos_rojos_en_roi, _ = cv2.findContours(mascara_rojo_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Para cada dado rojo encontrado, calcular coordenadas absolutas
        for contorno_rojo in contornos_rojos_en_roi:
            x_rojo, y_rojo, w_rojo, h_rojo = cv2.boundingRect(contorno_rojo)

            # Filtrar por área (dados muy pequeños o muy grandes)
            area = w_rojo * h_rojo
            if area < area_min_dado or area > area_max_dado:
                continue

            # Filtrar por aspect ratio (dados deben ser aproximadamente cuadrados)
            aspect_ratio = w_rojo / h_rojo
            if aspect_ratio < 0.8 or aspect_ratio > 1.2:
                continue

            # Convertir coordenadas relativas (dentro de ROI) a absolutas (frame completo)
            x_abs = x_verde + x_rojo
            y_abs = y_verde + y_rojo

            # Calcular centroide del dado rojo
            cx = x_abs + w_rojo // 2
            cy = y_abs + h_rojo // 2

            centroides.append((cx, cy))
            boxes.append((x_abs, y_abs, w_rojo, h_rojo))
            contornos_dados_rojos.append(contorno_rojo)

    return contornos_dados_rojos, frame_threshold_numero, centroides, boxes

def emparejar_centroides(centroides_actuales, centroides_previos, umbral=1):
    """
    Empareja centroides entre frames basándose en distancia euclidiana.

    Args:
        centroides_actuales: Lista de tuplas (cx, cy)
        centroides_previos: Lista de tuplas (cx, cy)
        umbral: Distancia máxima para considerar que es el mismo objeto

    Returns:
        pares_estaticos: Lista de tuplas (idx_actual, idx_prev) de objetos que NO se movieron
    """
    pares_estaticos = []

    for i, (cx_actual, cy_actual) in enumerate(centroides_actuales):
        for j, (cx_prev, cy_prev) in enumerate(centroides_previos):
            # Calcular distancia euclidiana
            distancia = math.sqrt((cx_actual - cx_prev)**2 + (cy_actual - cy_prev)**2)

            if distancia < umbral:
                pares_estaticos.append((i, j))
                break  # Un centroide actual solo se empareja con un previo

    return pares_estaticos

def procesar_roi_numeros(frame, frame_threshold_numero, x, y, w, h, copia_original, rois, rois_numeros_filtrados, rois_clausuras,nombre):
    """
    Procesa una ROI para detectar y contar números.

    Args:
        frame: Frame original BGR
        frame_threshold_numero: Máscara de números del frame completo
        x, y, w, h: Coordenadas del bounding box de la ROI
        copia_original: Imagen donde dibujar los resultados (modificada in-place)
        rois: Lista para almacenar ROIs extraídas (modificada in-place)
        rois_numeros_filtrados: Lista para almacenar ROIs filtradas (modificada in-place)
        rois_clausuras: Lista para almacenar ROIs con clausura aplicada (modificada in-place)

    Returns:
        num_contornos: Número de contornos detectados en la ROI
    """
    # Extraer ROI
    roi = frame[y:y+h, x:x+w]
    rois.append(roi)

    # # Recortar la máscara de números para que coincida con la ROI
    mascara_roi = frame_threshold_numero[y:y+h, x:x+w]

    # Aplicar apertura y clausura morfológica
    sa = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 2))
    s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    roi_apertura = cv2.morphologyEx(mascara_roi, cv2.MORPH_OPEN, sa, iterations=1)
    roi_clausura = cv2.morphologyEx(roi_apertura, cv2.MORPH_CLOSE, s, iterations=2)

    # Encontrar contornos internos (ahora sobre imagen binaria)
    _, _, _, centroides = cv2.connectedComponentsWithStats(roi_clausura, 8, cv2.CV_32S)

    num_contornos = len(centroides)-1

    # Dibujar en la imagen con el número de contornos
    cv2.rectangle(copia_original, (x, y), (x+w, y+h), (128, 0, 128), 2)
    cv2.putText(copia_original, str(num_contornos), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2)
    cv2.putText(copia_original, nombre, (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    rois_numeros_filtrados.append(num_contornos)
    rois_clausuras.append(roi_clausura)

    return num_contornos

def detectar_moda(tiradas):
    '''
    Calcula la moda de las tiradas de los dados
    
    Args:
        tiradas: lista de listas que contiene los valores del dado en cada frame

    Returns:
        modas_por_dado: diccionario del tipo {dado:moda}
    '''
    # Convertir la lista de listas a un array NumPy
    array_dados = np.array(tiradas)

    # Transponer el array
    array_dados_transpuesto = array_dados.T

    # Calcular la moda para cada dado (cada fila del array transpuesto)
    modas_por_dado = {}
    num_dados = array_dados_transpuesto.shape[0]

    nombres = ['Julio', 'Chuck', 'Dani', 'Noether','Ernesto']

    for i in range(num_dados):
        valores_dado = array_dados_transpuesto[i]

        # Contar la frecuencia de los valores de este dado
        frecuencias = Counter(valores_dado)

        # Para manejar empates:
        max_frecuencia = max(frecuencias.values())
        moda_actual = [
            valor for valor, cuenta in frecuencias.items() 
            if cuenta == max_frecuencia
        ]
        nombre = nombres[i]
        modas_por_dado[nombre] = moda_actual[0].item()

    return modas_por_dado