# Procesador de Videos de Dados

Sistema de detección y análisis automático de dados en videos, utilizando procesamiento de imágenes con OpenCV.

## Instrucciones de Uso

### 1. Preparar los Videos de Entrada

Coloca los videos `.mp4` que deseas procesar en la carpeta.

### 2. Ejecutar el Procesador

```bash
python procesamiento.py
```

### 3. Esperar el Procesamiento

El script procesará automáticamente todos los archivos `.mp4` encontrados en el directorio actual (excepto los que empiezan con `output_` para evitar reprocesar salidas previas).

Durante la ejecución verás:
```
Se encontraron N videos para procesar

Procesando: nombre_video

Julio: 5
Chuck: 3
Dani: 6
Noether: 4
Ernesto: 2

Video procesado: outputs/nombre_video.mp4
Frames procesados: 450

¡Procesamiento completo!
```

## Salidas Generadas

### Videos Procesados

Los videos procesados se guardan en la carpeta `outputs/` con el mismo nombre que el video original. Cada frame procesado incluye:

- **Rectángulos morados** alrededor de cada dado detectado
- **Número de puntos** detectados en cada dado (arriba del rectángulo)
- **Nombre del dado** (debajo del número)

### Información en Consola

Para cada video procesado, el script imprime:

1. **Nombre del video** siendo procesado
2. **Resultado de la moda** para cada dado (valor más frecuente detectado a lo largo del video)
3. **Ruta del video de salida**
4. **Número total de frames** procesados
