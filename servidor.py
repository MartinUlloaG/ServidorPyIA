import math
import tempfile
import os
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

app = FastAPI()

MODEL_PATH = "pose_landmarker_full.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1n = v1 / (np.linalg.norm(v1) + 1e-9)
    v2n = v2 / (np.linalg.norm(v2) + 1e-9)
    dot = float(np.clip(np.dot(v1n, v2n), -1.0, 1.0))
    return math.degrees(math.acos(dot))

def joint_angle(a, b, c) -> float:
    return angle_between(a - b, c - b)

def lm_to_np(lm) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

# ---------- Índices ----------
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24
L_KNE, R_KNE = 25, 26
L_ANK, R_ANK = 27, 28
L_ELB, R_ELB = 13, 14
L_WRI, R_WRI = 15, 16

SQUAT_KNEE_ASYM_THRESHOLD = 14.0
SQUAT_KNEE_TRACK_THRESHOLD = 0.10
LUNGE_FRONT_KNEE_MAX = 125.0
LUNGE_REAR_KNEE_MIN = 130.0
LUNGE_STEP_WIDTH_MIN = 0.10
CURL_FLEXED_MAX = 75.0
CURL_EXTENDED_MIN = 150.0
CURL_SYMMETRY_THRESHOLD = 18.0
CURL_ELBOW_DRIFT_THRESHOLD = 0.12
PRESS_UP_ELBOW_MIN = 155.0
PRESS_DOWN_ELBOW_MAX = 95.0
PRESS_WRIST_ABOVE_SHOULDER = 0.02
PRESS_WRIST_NEAR_SHOULDER = 0.10
PRESS_SYMMETRY_THRESHOLD = 16.0

EJERCICIOS = ["sentadilla", "zancada", "curl_biceps_sentado", "press_hombros_sentado"]

def analizar_frame(pose_lms, ejercicio: str):
    feedback = []
    metricas = {}

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

    left_knee_angle  = joint_angle(l_hip, l_kne, l_ank)
    right_knee_angle = joint_angle(r_hip, r_kne, r_ank)
    avg_knee_angle   = (left_knee_angle + right_knee_angle) / 2.0
    knee_angle_diff  = abs(left_knee_angle - right_knee_angle)

    left_elbow_angle  = joint_angle(l_sho, l_elb, l_wri)
    right_elbow_angle = joint_angle(r_sho, r_elb, r_wri)
    avg_elbow_angle   = (left_elbow_angle + right_elbow_angle) / 2.0

    if ejercicio == "sentadilla":
        metricas["rodilla_izq"] = round(left_knee_angle, 1)
        metricas["rodilla_der"] = round(right_knee_angle, 1)
        metricas["promedio_rodillas"] = round(avg_knee_angle, 1)

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

        if knee_angle_diff > SQUAT_KNEE_ASYM_THRESHOLD:
            feedback.append("Sentadilla: distribuya mejor el peso entre ambas piernas.")

        left_knee_track  = abs(l_kne[0] - l_ank[0])
        right_knee_track = abs(r_kne[0] - r_ank[0])
        if left_knee_track > SQUAT_KNEE_TRACK_THRESHOLD:
            feedback.append("Tobillo/Rodilla I: alinee la rodilla con el pie.")
        if right_knee_track > SQUAT_KNEE_TRACK_THRESHOLD:
            feedback.append("Tobillo/Rodilla D: alinee la rodilla con el pie.")

    elif ejercicio == "zancada":
        if left_knee_angle <= right_knee_angle:
            front_knee_angle = left_knee_angle
            rear_knee_angle  = right_knee_angle
            front_knee_track = abs(l_kne[0] - l_ank[0])
        else:
            front_knee_angle = right_knee_angle
            rear_knee_angle  = left_knee_angle
            front_knee_track = abs(r_kne[0] - r_ank[0])

        step_width = abs(l_ank[0] - r_ank[0])
        metricas["rodilla_delantera"] = round(front_knee_angle, 1)
        metricas["rodilla_trasera"]   = round(rear_knee_angle, 1)
        metricas["separacion_tobillos"] = round(step_width, 3)

        if front_knee_angle > LUNGE_FRONT_KNEE_MAX:
            feedback.append("Rodilla delantera: flexione un poco mas.")
        else:
            feedback.append("Rodilla delantera: buena flexion.")

        if rear_knee_angle < LUNGE_REAR_KNEE_MIN:
            feedback.append("Rodilla trasera: extienda un poco mas.")
        if step_width < LUNGE_STEP_WIDTH_MIN:
            feedback.append("Tobillos: aumente separacion entre pies.")
        if front_knee_track > 0.12:
            feedback.append("Rodilla/Tobillo delantero: mejore alineacion.")

    elif ejercicio == "curl_biceps_sentado":
        elbow_asym       = abs(left_elbow_angle - right_elbow_angle)
        left_elbow_drift = abs(l_elb[0] - l_sho[0])
        right_elbow_drift= abs(r_elb[0] - r_sho[0])

        metricas["codo_izq"] = round(left_elbow_angle, 1)
        metricas["codo_der"] = round(right_elbow_angle, 1)
        metricas["promedio_codos"] = round(avg_elbow_angle, 1)

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

    elif ejercicio == "press_hombros_sentado":
        elbow_asym       = abs(left_elbow_angle - right_elbow_angle)
        left_wrist_above = l_sho[1] - l_wri[1]
        right_wrist_above= r_sho[1] - r_wri[1]
        shoulder_y_diff  = abs(l_sho[1] - r_sho[1])

        metricas["codo_izq"] = round(left_elbow_angle, 1)
        metricas["codo_der"] = round(right_elbow_angle, 1)

        if avg_elbow_angle < 140.0:
            feedback.append("Press hombros: extienda mas en la parte alta.")
        else:
            feedback.append("Press hombros: buena extension general.")

        if elbow_asym > PRESS_SYMMETRY_THRESHOLD:
            feedback.append("Press hombros: mejore la simetria entre ambos brazos.")
        if shoulder_y_diff > 0.03:
            feedback.append("Press hombros: nivele los hombros durante el empuje.")

    return feedback, metricas


def calcular_puntaje(feedbacks_totales: list) -> int:
    negativos = sum(1 for f in feedbacks_totales if any(
        w in f.lower() for w in ["mejore", "alinee", "flexione", "extienda", "nivele", "acerque", "aumente", "distribuya", "baje", "cierre"]
    ))
    total = len(feedbacks_totales) if feedbacks_totales else 1
    ratio = negativos / total
    return max(0, int(100 - ratio * 100))


@app.post("/analizar")
async def analizar_video(
    video: UploadFile = File(...),
    ejercicio: str = Form("sentadilla")
):
    if ejercicio not in EJERCICIOS:
        return JSONResponse(status_code=400, content={"error": f"Ejercicio no válido. Opciones: {EJERCICIOS}"})

    # Guardar video temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    todos_feedback = []
    ultimas_metricas = {}

    try:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_poses=1
        )

        with PoseLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_idx = 0
            sample_rate = max(1, int(fps / 5))  # analiza 5 frames por segundo

            while cap.isOpened():
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                if frame_idx % sample_rate == 0:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    timestamp_ms = int((frame_idx / fps) * 1000)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)

                    if result.pose_landmarks:
                        pose_lms = result.pose_landmarks[0]
                        fb, met  = analizar_frame(pose_lms, ejercicio)
                        todos_feedback.extend(fb)
                        ultimas_metricas = met

                frame_idx += 1

            cap.release()

    finally:
        os.unlink(tmp_path)

    feedback_unico = list(dict.fromkeys(todos_feedback))
    puntaje = calcular_puntaje(todos_feedback)

    # Convertir numpy types a Python nativos para serialización
    metricas_limpias = {k: float(v) for k, v in ultimas_metricas.items()}

    return {
        "ejercicio": ejercicio,
        "puntaje": puntaje,
        "feedback": feedback_unico,
        "metricas": metricas_limpias
    }


@app.get("/")
def root():
    return {"status": "Servidor MediaPipe corriendo"}