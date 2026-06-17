# Proyecto de Feedback de Pose para Fútbol

Este proyecto utiliza MediaPipe Pose Landmarker para detectar la pose del cuerpo y proporcionar feedback en tiempo real sobre la postura, enfocándose en el tren superior (tronco, hombros, codos).

## Requisitos

- Python 3.10+ (recomendado 3.12)
- Webcam funcional
- Sistema operativo: Windows, Linux o macOS

## Instalación Paso a Paso

1. **Clona el repositorio:**
   ```
   git clone https://github.com/EstebanSalgad0/ProyectoMovimiento-.git
   cd ProyectoMovimiento-
   ```

2. **Crea un entorno virtual:**
   - En Windows (PowerShell o CMD):
     ```
     python -m venv venv
     ```
   - En Linux/macOS:
     ```
     python3 -m venv venv
     ```

3. **Activa el entorno virtual:**
   - En Windows:
     ```
     venv\Scripts\activate
     ```
     (Deberías ver `(venv)` al inicio del prompt)
   - En Linux/macOS:
     ```
     source venv/bin/activate
     ```

4. **Instala las dependencias:**
   ```
   pip install mediapipe opencv-python numpy
   ```
   Esto instalará MediaPipe, OpenCV y NumPy. Si hay problemas con NumPy en Python 3.14, instala una versión anterior:
   ```
   pip install "numpy<2.3"
   ```

5. **Verifica la instalación (opcional):**
   ```
   python -c "import cv2; import mediapipe; import numpy as np; print('Instalacion exitosa')"
   ```

## Uso

1. **Ejecuta el script:**
   ```
   python pose_feedback_webcam.py
   ```

2. **Interacción:**
   - Se abrirá una ventana con la webcam y el esqueleto dibujado en verde.
   - Las métricas aparecen en la esquina superior izquierda.
   - El feedback en la esquina inferior derecha.
   - Presiona **1** para modo **sentadilla**.
   - Presiona **2** para modo **zancada**.
   - Presiona **3** para modo **curl de biceps sentado**.
   - Presiona **4** para modo **press de hombros sentado**.
   - Presiona **ESC** para salir.

3. **Pruebas:**
   - Inclina el tronco lateralmente.
   - Flexiona los codos.
   - Sube un hombro.
   - En sentadilla: baja y sube, verifica profundidad y simetria de rodillas.
   - En zancada: adelanta una pierna, verifica flexion delantera y separacion de pies.
   - En curl sentado: mantente sentado, flexiona y extiende codos de forma controlada.
   - En press sentado: eleva brazos sobre hombros y vuelve a la posicion inicial.
   - Observa cómo cambian las métricas y el feedback.

## Solución de Problemas

- **La ventana no se abre o se cierra inmediatamente:**
  - Asegúrate de que no haya otras aplicaciones usando la webcam (cierra Zoom, Discord, etc.).
  - Ejecuta en PowerShell externo (no en VS Code integrado).
  - Si persiste, usa un archivo de video: cambia `cap = cv2.VideoCapture(0)` por `cap = cv2.VideoCapture("tu_video.mp4")`.

- **Errores de codificación en texto:**
  - El script usa texto sin acentos para evitar problemas.

- **Problemas con la webcam:**
  - Verifica permisos en Configuración > Privacidad > Cámara.
  - Actualiza drivers de la webcam.
   - Si toma otra camara (virtual), fuerza el indice con variable de entorno:
      - PowerShell: `$env:CAMERA_INDEX=1` y luego `python pose_feedback_webcam.py`
      - CMD: `set CAMERA_INDEX=1` y luego `python pose_feedback_webcam.py`

## Archivos Incluidos

- `pose_feedback_webcam.py`: Script principal.
- `pose_landmarker_lite.task`: Modelo rápido (recomendado).
- `pose_landmarker_full.task`: Modelo preciso (más lento).
- `.gitignore`: Ignora entornos virtuales.
- `README.md`: Esta documentación.

## Notas

- El proyecto se enfoca en métricas del tren superior.
- Tambien incluye reglas basicas de tren inferior para sentadilla y zancada.
- Incluye ejercicios de tren superior para test sentado: curl de biceps y press de hombros.
- Los umbrales se pueden ajustar en el código para personalizar el feedback.
- Para desarrollo, usa el modelo lite; para producción, el full.