# Este procesa el video directamente

import os
import cv2
import glob
from utils import procesar_frame, procesar_roi_numeros, emparejar_centroides, detectar_moda

# Crear carpeta de salida
os.makedirs("outputs", exist_ok=True)

# --- Carpeta con videos ------------------------------------------------
carpeta_videos = './'
patron_videos = os.path.join(carpeta_videos, '*.mp4')
videos = glob.glob(patron_videos)

print(f"Se encontraron {len(videos)} videos para procesar")

nombres = ['Julio', 'Chuck', 'Dani', 'Noether','Ernesto']

# Procesar cada video
for video_path in videos:
    # Obtener nombre del archivo sin extensión
    nombre_archivo = os.path.splitext(os.path.basename(video_path))[0]

    # Saltar videos de salida (que ya fueron procesados)
    if nombre_archivo.startswith('output_'):
        print(f"Saltando {nombre_archivo} (es un archivo de salida)")
        continue

    print(f"\nProcesando: {nombre_archivo}\n")

    # --- Leer el video ------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Crear VideoWriter para guardar el video procesado en la carpeta outputs
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join('outputs', f'{nombre_archivo}.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(width/3), int(height/3)))

    frame_number = 0
    prev_frame = None
    prev_contornos = None
    prev_frame_threshold_numero = None
    prev_centroides = None
    prev_boxes = None

    rois = []
    rois_numeros_filtrados = []
    rois_clausuras = []

    tiradas=[]

    while cap.isOpened():
        ret, frame = cap.read()
        nombre = 0

        if ret == True:
            frame = cv2.resize(frame, dsize=(int(width/3), int(height/3)))

            contornos, frame_threshold_numero, centroides, boxes = procesar_frame(frame)

            copia_original = frame.copy()

            if prev_centroides is not None and len(centroides) > 0:
                # Encontrar objetos estáticos
                pares_estaticos = emparejar_centroides(centroides, prev_centroides, umbral=10)
                
                tirada_frame=[] #lista que guarda los valores de los dados en ese frame

                # Procesar solo ROIs estáticas
                for idx_actual, idx_prev in pares_estaticos:
                    x, y, w, h = boxes[idx_actual]

                    # Procesar ROI para detectar números
                    num_contornos = procesar_roi_numeros(
                        frame, frame_threshold_numero, x, y, w, h,
                        copia_original, rois, rois_numeros_filtrados, rois_clausuras, nombres[nombre]
                    )
                    tirada_frame.append(num_contornos) 
                    nombre+=1

                if len(tirada_frame)==5:
                    tiradas.append(tirada_frame) #solo añade las tiradas que detectan los 5 dados
                #print(tiradas)
                #print('--')

            # Escribir frame procesado al video de salida
            out.write(copia_original)


            frame_number += 1

            # Guardar el frame actual como el frame anterior para la próxima iteración
            prev_frame = frame.copy()
            prev_contornos = contornos
            prev_frame_threshold_numero = frame_threshold_numero
            prev_centroides = centroides
            prev_boxes = boxes

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
        if nombre<5:
            nombre+=1
    
    valores = detectar_moda(tiradas)
    cap.release()
    out.release()

    for clave, valor in valores.items():
        print(f"{clave}: {valor}")

    print(f"\nVideo procesado: {output_path}")
    print(f"Frames procesados: {frame_number}")

cv2.destroyAllWindows()
print("\n¡Procesamiento completo!")
