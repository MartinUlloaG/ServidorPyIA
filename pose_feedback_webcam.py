import time
import math
import os
import cv2
import numpy as np
import mediapipe as mp

MODEL_PATH = "pose_landmarker_full.task"
CAMERA_SCAN_MAX_INDEX = 10
WINDOW_NAME = "Pose Feedback (MediaPipe)"

camera_index_from_env = os.getenv("CAMERA_INDEX", "1")
try:
    PREFERRED_CAMERA_INDEX = int(camera_index_from_env)
except ValueError:
    PREFERRED_CAMERA_INDEX = 1

# ---------- Utilidades matemáticas ----------
def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Ángulo (grados) entre dos vectores."""
    v1n = v1 / (np.linalg.norm(v1) + 1e-9)
    v2n = v2 / (np.linalg.norm(v2) + 1e-9)
    dot = float(np.clip(np.dot(v1n, v2n), -1.0, 1.0))
    return math.degrees(math.acos(dot))

def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Ángulo (grados) en la articulación b, formado por a-b-c."""
    ba = a - b
    bc = c - b
    return angle_between(ba, bc)

def lm_to_np(lm) -> np.ndarray:
    """Landmark normalizado (x,y,z) a numpy."""
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

def midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return (p1 + p2) / 2.0

# ---------- MediaPipe Tasks ----------
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_poses=1
)

# Índices de landmarks (MediaPipe Pose: 33 puntos)
# 11: left_shoulder, 12: right_shoulder
# 23: left_hip, 24: right_hip
# 25: left_knee, 26: right_knee
# 27: left_ankle, 28: right_ankle
# 13: left_elbow, 14: right_elbow
# 15: left_wrist, 16: right_wrist
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24
L_KNE, R_KNE = 25, 26
L_ANK, R_ANK = 27, 28
L_ELB, R_ELB = 13, 14
L_WRI, R_WRI = 15, 16

# Conexiones para dibujar el esqueleto
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (25, 26), (27, 28), (29, 30), (31, 32),
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)
]

UPPER_BODY_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

LOWER_BODY_CONNECTIONS = [
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (25, 26), (27, 28),
]

# Umbrales (ajustables)
TRUNK_TILT_DEG_THRESHOLD = 12.0      # tronco inclinado (lateral) sobre ~12°
KNEE_FLEXION_THRESHOLD = 160.0       # rodilla < 160° => flexión notoria
SHOULDER_ASYM_Y_THRESHOLD = 0.03     # diferencia en y normalizada (~3% altura)

# Modos de ejercicio y umbrales de tren inferior
EXERCISE_SQUAT = "sentadilla"
EXERCISE_LUNGE = "zancada"
EXERCISE_BICEPS_CURL = "curl_biceps_sentado"
EXERCISE_SHOULDER_PRESS = "press_hombros_sentado"

SQUAT_DOWN_ANGLE = 115.0              # menor a este valor => fase abajo
SQUAT_UP_ANGLE = 160.0                # mayor a este valor => fase arriba
SQUAT_KNEE_ASYM_THRESHOLD = 14.0      # diferencia entre rodillas
SQUAT_KNEE_TRACK_THRESHOLD = 0.10     # desviacion lateral rodilla-tobillo (normalizada)

LUNGE_FRONT_KNEE_MAX = 125.0          # pierna delantera debe flexionar
LUNGE_REAR_KNEE_MIN = 130.0           # pierna trasera debe quedar mas extendida
LUNGE_STEP_WIDTH_MIN = 0.10           # separacion horizontal minima entre tobillos
LUNGE_KNEE_TRACK_THRESHOLD = 0.12     # desviacion lateral rodilla-tobillo

# Umbrales de tren superior para ejercicios sentado
CURL_FLEXED_MAX = 75.0                # codo por debajo de este angulo => fase arriba
CURL_EXTENDED_MIN = 150.0             # codo por encima de este angulo => fase abajo
CURL_SYMMETRY_THRESHOLD = 18.0        # diferencia maxima entre codos
CURL_ELBOW_DRIFT_THRESHOLD = 0.12     # desplazamiento lateral de codo respecto hombro

PRESS_UP_ELBOW_MIN = 155.0            # codo extendido en la parte alta
PRESS_DOWN_ELBOW_MAX = 95.0           # codo flexionado en la parte baja
PRESS_WRIST_ABOVE_SHOULDER = 0.02     # muneca claramente por encima del hombro
PRESS_WRIST_NEAR_SHOULDER = 0.10      # muneca cerca de nivel de hombro
PRESS_SYMMETRY_THRESHOLD = 16.0       # diferencia maxima entre codos

exercise_mode = EXERCISE_SQUAT

MODE_FOCUS_LANDMARKS = {
    EXERCISE_SQUAT: [L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK],
    EXERCISE_LUNGE: [L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK],
    EXERCISE_BICEPS_CURL: [L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI],
    EXERCISE_SHOULDER_PRESS: [L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI],
}

LANDMARK_LABELS = {
    L_SHO: "Hombro I",
    R_SHO: "Hombro D",
    L_ELB: "Codo I",
    R_ELB: "Codo D",
    L_WRI: "Muneca I",
    R_WRI: "Muneca D",
    L_HIP: "Cadera I",
    R_HIP: "Cadera D",
    L_KNE: "Rodilla I",
    R_KNE: "Rodilla D",
    L_ANK: "Tobillo I",
    R_ANK: "Tobillo D",
}

def get_mode_connections(mode: str):
    if mode in (EXERCISE_SQUAT, EXERCISE_LUNGE):
        return LOWER_BODY_CONNECTIONS
    return UPPER_BODY_CONNECTIONS

def put_lines(frame, lines, x=20, y0=30, dy=28, font_scale=0.85):
    y = y0
    for s in lines:
        cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        y += dy

def wrap_text_to_width(text: str, max_width: int, font_scale: float = 0.78, thickness: int = 2):
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        candidate_w = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
        if candidate_w <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines

def draw_text_panel(frame, title, lines, x, y, width, line_h=30, font_scale=0.78):
    padding = 12
    wrapped = []
    for line in lines:
        wrapped.extend(wrap_text_to_width(line, width - (padding * 2), font_scale=font_scale, thickness=2))

    max_lines = max(4, (frame.shape[0] - y - 20 - (padding * 2) - line_h) // line_h)
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines - 1] + ["..."]

    panel_h = (padding * 2) + line_h + (line_h * len(wrapped))
    x2 = min(frame.shape[1] - 10, x + width)
    y2 = min(frame.shape[0] - 10, y + panel_h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.rectangle(frame, (x, y), (x2, y2), (255, 255, 255), 1)

    cv2.putText(frame, title, (x + padding, y + padding + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    y_cursor = y + padding + line_h + 6
    for line in wrapped:
        cv2.putText(frame, line, (x + padding, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        y_cursor += line_h

def draw_focus_landmarks(frame, pose_lms, landmark_ids, color=(0, 220, 255)):
    height, width = frame.shape[:2]
    for idx in landmark_ids:
        if idx < len(pose_lms):
            lm = pose_lms[idx]
            pt = (int(lm.x * width), int(lm.y * height))
            cv2.circle(frame, pt, 6, color, -1)
            label = LANDMARK_LABELS.get(idx, f"lm{idx}")
            cv2.putText(frame, label, (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def open_camera_with_fallback(preferred_index: int, scan_max_index: int = 10):
    """Abre camara con indice preferido y fallback por escaneo."""
    def try_open(index: int):
        # En Windows se intenta primero DSHOW y luego backend automatico.
        for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
            cap_try = cv2.VideoCapture(index, backend)
            if cap_try.isOpened():
                valid_reads = 0
                for _ in range(3):
                    ok, frame = cap_try.read()
                    if ok and frame is not None and len(frame.shape) == 3 and frame.shape[0] > 0 and frame.shape[1] > 0:
                        valid_reads += 1
                if valid_reads >= 2:
                    return cap_try
            cap_try.release()
        return None

    cap = try_open(preferred_index)
    if cap is not None:
        return cap, preferred_index

    # Fallback: escanear indices para encontrar una camara que entregue frame.
    for idx in range(scan_max_index + 1):
        cap = try_open(idx)
        if cap is not None:
            return cap, idx

    raise RuntimeError(
        "No se pudo abrir ninguna camara. Pruebe setear CAMERA_INDEX (ej: 0,1,2)."
    )

start_time = time.time()

with PoseLandmarker.create_from_options(options) as landmarker:
    cap, camera_index_in_use = open_camera_with_fallback(
        PREFERRED_CAMERA_INDEX,
        CAMERA_SCAN_MAX_INDEX
    )
    print(f"Camara activa: indice {camera_index_in_use}")

    # Opcional: setear propiedades antes de leer
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1600, 900)
    try:
        while cap.isOpened():
            try:
                ok, frame_bgr = cap.read()
            except cv2.error:
                print("Error de lectura de camara. Intente otro indice con CAMERA_INDEX.")
                break
            if not ok:
                print("cap.read() fallo")
                break
            if frame_bgr is None or len(frame_bgr.shape) != 3:
                print("Frame invalido recibido de la camara")
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            feedback = []
            metrics = []

            # Dibujar pose si existe
            if result.pose_landmarks:
                pose_lms = result.pose_landmarks[0]

                # Dibujo segmentado del esqueleto segun ejercicio
                height, width = frame_bgr.shape[:2]
                mode_connections = get_mode_connections(exercise_mode)
                for start_idx, end_idx in mode_connections:
                    if start_idx < len(pose_lms) and end_idx < len(pose_lms):
                        start_lm = pose_lms[start_idx]
                        end_lm = pose_lms[end_idx]
                        start_point = (int(start_lm.x * width), int(start_lm.y * height))
                        end_point = (int(end_lm.x * width), int(end_lm.y * height))
                        cv2.line(frame_bgr, start_point, end_point, (0, 255, 0), 2)

                draw_focus_landmarks(frame_bgr, pose_lms, MODE_FOCUS_LANDMARKS.get(exercise_mode, []))

                # ---- Extraer puntos relevantes ----
                l_sho = lm_to_np(pose_lms[L_SHO])
                r_sho = lm_to_np(pose_lms[R_SHO])
                l_hip = lm_to_np(pose_lms[L_HIP])
                r_hip = lm_to_np(pose_lms[R_HIP])
                l_elb = lm_to_np(pose_lms[L_ELB])
                r_elb = lm_to_np(pose_lms[R_ELB])
                l_wri = lm_to_np(pose_lms[L_WRI])
                r_wri = lm_to_np(pose_lms[R_WRI])
                l_kne = lm_to_np(pose_lms[L_KNE])
                r_kne = lm_to_np(pose_lms[R_KNE])
                l_ank = lm_to_np(pose_lms[L_ANK])
                r_ank = lm_to_np(pose_lms[R_ANK])

                # Angulos de codo (solo se muestran en modos de tren superior)
                left_elbow_angle = joint_angle(l_sho, l_elb, l_wri)
                right_elbow_angle = joint_angle(r_sho, r_elb, r_wri)

                # Asimetria de hombros (usada en press)
                shoulder_y_diff = float(abs(l_sho[1] - r_sho[1]))  # y normalizada

                # ---- Reglas por ejercicio (feedback segmentado) ----
                left_knee_angle = joint_angle(l_hip, l_kne, l_ank)
                right_knee_angle = joint_angle(r_hip, r_kne, r_ank)
                avg_knee_angle = (left_knee_angle + right_knee_angle) / 2.0
                knee_angle_diff = abs(left_knee_angle - right_knee_angle)

                metrics.append(f"Ejercicio: {exercise_mode}")

                if exercise_mode == EXERCISE_SQUAT:
                    points_used = "Puntos: Cadera(23/24), Rodilla(25/26), Tobillo(27/28)"
                    metrics.append(points_used)
                    metrics.append(f"Rodilla izq: {left_knee_angle:.1f} deg")
                    metrics.append(f"Rodilla der: {right_knee_angle:.1f} deg")
                    metrics.append(f"Promedio rodillas: {avg_knee_angle:.1f} deg")

                    hip_diff = abs(l_hip[1] - r_hip[1])
                    if hip_diff > 0.04:
                        feedback.append("Cadera: nivele la pelvis para mayor estabilidad.")
                    else:
                        feedback.append("Cadera: pelvis estable.")

                    if avg_knee_angle > 135.0:
                        feedback.append("Rodillas: baje un poco mas para activar piernas.")
                    elif avg_knee_angle < 80.0:
                        feedback.append("Rodillas: profundidad alta, controle el descenso.")
                    else:
                        feedback.append("Rodillas: buena profundidad.")

                    if avg_knee_angle < SQUAT_DOWN_ANGLE:
                        feedback.append("Fase sentadilla: abajo.")
                    elif avg_knee_angle > SQUAT_UP_ANGLE:
                        feedback.append("Fase sentadilla: arriba.")
                    else:
                        feedback.append("Fase sentadilla: transicion.")

                    if knee_angle_diff > SQUAT_KNEE_ASYM_THRESHOLD:
                        feedback.append("Sentadilla: distribuya mejor el peso entre ambas piernas.")

                    left_knee_track = abs(l_kne[0] - l_ank[0])
                    right_knee_track = abs(r_kne[0] - r_ank[0])
                    metrics.append(f"Alineacion rodilla izq (Dx): {left_knee_track:.3f}")
                    metrics.append(f"Alineacion rodilla der (Dx): {right_knee_track:.3f}")

                    if left_knee_track > SQUAT_KNEE_TRACK_THRESHOLD:
                        feedback.append("Tobillo/Rodilla I: alinee la rodilla con el pie.")
                    if right_knee_track > SQUAT_KNEE_TRACK_THRESHOLD:
                        feedback.append("Tobillo/Rodilla D: alinee la rodilla con el pie.")

                elif exercise_mode == EXERCISE_LUNGE:
                    points_used = "Puntos: Cadera(23/24), Rodilla(25/26), Tobillo(27/28)"
                    metrics.append(points_used)
                    # Se considera delantera la pierna con mayor flexion (angulo menor)
                    if left_knee_angle <= right_knee_angle:
                        front_leg = "izquierda"
                        front_knee_angle = left_knee_angle
                        rear_knee_angle = right_knee_angle
                        front_knee_track = abs(l_kne[0] - l_ank[0])
                    else:
                        front_leg = "derecha"
                        front_knee_angle = right_knee_angle
                        rear_knee_angle = left_knee_angle
                        front_knee_track = abs(r_kne[0] - r_ank[0])

                    step_width = abs(l_ank[0] - r_ank[0])
                    metrics.append(f"Pierna delantera: {front_leg}")
                    metrics.append(f"Rodilla delantera: {front_knee_angle:.1f} deg")
                    metrics.append(f"Rodilla trasera: {rear_knee_angle:.1f} deg")
                    metrics.append(f"Separacion tobillos (Dx): {step_width:.3f}")

                    if front_knee_angle > LUNGE_FRONT_KNEE_MAX:
                        feedback.append("Rodilla delantera: flexione un poco mas.")
                    else:
                        feedback.append("Rodilla delantera: buena flexion.")

                    if rear_knee_angle < LUNGE_REAR_KNEE_MIN:
                        feedback.append("Rodilla trasera: extienda un poco mas.")

                    if step_width < LUNGE_STEP_WIDTH_MIN:
                        feedback.append("Tobillos: aumente separacion entre pies.")

                    if front_knee_track > LUNGE_KNEE_TRACK_THRESHOLD:
                        feedback.append("Rodilla/Tobillo delantero: mejore alineacion.")

                    if front_knee_angle <= LUNGE_FRONT_KNEE_MAX and rear_knee_angle >= LUNGE_REAR_KNEE_MIN and step_width >= LUNGE_STEP_WIDTH_MIN:
                        feedback.append("Zancada: ejecucion estable.")

                elif exercise_mode == EXERCISE_BICEPS_CURL:
                    points_used = "Puntos: Hombro(11/12), Codo(13/14), Muneca(15/16)"
                    metrics.append(points_used)
                    avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2.0
                    elbow_asym = abs(left_elbow_angle - right_elbow_angle)
                    left_elbow_drift = abs(l_elb[0] - l_sho[0])
                    right_elbow_drift = abs(r_elb[0] - r_sho[0])

                    metrics.append(f"Codo izq: {left_elbow_angle:.1f} deg")
                    metrics.append(f"Codo der: {right_elbow_angle:.1f} deg")
                    metrics.append(f"Promedio codos: {avg_elbow_angle:.1f} deg")
                    metrics.append(f"Deriva codo izq (Dx): {left_elbow_drift:.3f}")
                    metrics.append(f"Deriva codo der (Dx): {right_elbow_drift:.3f}")

                    if avg_elbow_angle < CURL_FLEXED_MAX:
                        feedback.append("Curl: fase arriba (contraccion).")
                    elif avg_elbow_angle > CURL_EXTENDED_MIN:
                        feedback.append("Curl: fase abajo (extension).")
                    else:
                        feedback.append("Curl: fase media.")

                    if avg_elbow_angle > 120.0:
                        feedback.append("Codo: flexione mas para completar el curl.")
                    elif avg_elbow_angle < 55.0:
                        feedback.append("Codo: no cierre tanto, controle el movimiento.")
                    else:
                        feedback.append("Codo: rango de movimiento adecuado.")

                    if elbow_asym > CURL_SYMMETRY_THRESHOLD:
                        feedback.append("Hombros/Codos: mejore simetria entre ambos brazos.")

                    if left_elbow_drift > CURL_ELBOW_DRIFT_THRESHOLD:
                        feedback.append("Hombro/Codo I: acerque el codo al torso.")
                    if right_elbow_drift > CURL_ELBOW_DRIFT_THRESHOLD:
                        feedback.append("Hombro/Codo D: acerque el codo al torso.")

                    left_wrist_to_elbow = abs(l_wri[0] - l_elb[0])
                    right_wrist_to_elbow = abs(r_wri[0] - r_elb[0])
                    if left_wrist_to_elbow > 0.22:
                        feedback.append("Muneca I: evite desviacion lateral excesiva.")
                    if right_wrist_to_elbow > 0.22:
                        feedback.append("Muneca D: evite desviacion lateral excesiva.")

                elif exercise_mode == EXERCISE_SHOULDER_PRESS:
                    points_used = "Puntos: Hombro(11/12), Codo(13/14), Muneca(15/16)"
                    metrics.append(points_used)
                    avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2.0
                    elbow_asym = abs(left_elbow_angle - right_elbow_angle)
                    left_wrist_above = (l_sho[1] - l_wri[1])
                    right_wrist_above = (r_sho[1] - r_wri[1])

                    metrics.append(f"Codo izq: {left_elbow_angle:.1f} deg")
                    metrics.append(f"Codo der: {right_elbow_angle:.1f} deg")
                    metrics.append(f"Promedio codos: {avg_elbow_angle:.1f} deg")
                    metrics.append(f"Muneca izq sobre hombro (Dy): {left_wrist_above:.3f}")
                    metrics.append(f"Muneca der sobre hombro (Dy): {right_wrist_above:.3f}")

                    is_up = (
                        left_elbow_angle > PRESS_UP_ELBOW_MIN and
                        right_elbow_angle > PRESS_UP_ELBOW_MIN and
                        left_wrist_above > PRESS_WRIST_ABOVE_SHOULDER and
                        right_wrist_above > PRESS_WRIST_ABOVE_SHOULDER
                    )
                    is_down = (
                        left_elbow_angle < PRESS_DOWN_ELBOW_MAX and
                        right_elbow_angle < PRESS_DOWN_ELBOW_MAX and
                        abs(left_wrist_above) < PRESS_WRIST_NEAR_SHOULDER and
                        abs(right_wrist_above) < PRESS_WRIST_NEAR_SHOULDER
                    )

                    if is_up:
                        feedback.append("Press hombros: fase arriba.")
                    elif is_down:
                        feedback.append("Press hombros: fase abajo.")
                    else:
                        feedback.append("Press hombros: fase intermedia.")

                    if not is_up and avg_elbow_angle < 140.0:
                        feedback.append("Press hombros: extienda mas en la parte alta.")
                    else:
                        feedback.append("Press hombros: buena extension general.")

                    if elbow_asym > PRESS_SYMMETRY_THRESHOLD:
                        feedback.append("Press hombros: mejore la simetria entre ambos brazos.")

                    if shoulder_y_diff > SHOULDER_ASYM_Y_THRESHOLD:
                        feedback.append("Press hombros: nivele los hombros durante el empuje.")

                # Limitar para evitar saturacion visual
                metrics = metrics[:8]
                feedback = feedback[:8]
            else:
                feedback.append("No se detecta la pose. Acerque el cuerpo a camara y mejore la iluminacion.")

            # Mostrar metricas + feedback en paneles para evitar texto cortado
            height, width = frame_bgr.shape[:2]
            info_lines = [
                f"Camara idx: {camera_index_in_use}",
                "Modo: 1-sentadilla 2-zancada 3-curl 4-press",
            ] + metrics
            draw_text_panel(frame_bgr, "METRICAS", info_lines, x=16, y=16, width=min(620, width // 2))
            draw_text_panel(frame_bgr, "FEEDBACK", feedback, x=max(16, width - 640), y=16, width=min(620, width - 32))

            cv2.imshow(WINDOW_NAME, frame_bgr)

            # Cerrar cuando el usuario presiona la X de la ventana
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('1'):
                exercise_mode = EXERCISE_SQUAT
            elif key == ord('2'):
                exercise_mode = EXERCISE_LUNGE
            elif key == ord('3'):
                exercise_mode = EXERCISE_BICEPS_CURL
            elif key == ord('4'):
                exercise_mode = EXERCISE_SHOULDER_PRESS
            if key == 27:  # ESC
                break
    except KeyboardInterrupt:
        print("Cierre por teclado")

cap.release()
cv2.destroyAllWindows()