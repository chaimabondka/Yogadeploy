# streamlit_app.py
import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import mediapipe as mp
import cv2
import requests

# --- Configuration Streamlit ---
st.set_page_config(
    page_title="Analyse de Postures de Yoga",
    page_icon="🧘‍♀️",
    layout="wide"
)

# --- Charger le modèle MobileNetV2 personnalisé ---
@st.cache_resource
def load_classification_model():
    path = os.path.join(os.getcwd(), "mobilenetv2_yoga_postures.keras")
    if not os.path.exists(path):
        st.error("Modèle introuvable: mobilenetv2_yoga_postures.keras")
        st.stop()
        
    try:
        return tf.keras.models.load_model(path,
        custom_objects={
            'Functional': TF_Functional
        })
    except Exception as e:
        st.error(f"Erreur de chargement du modèle : {type(e).__name__} - {e}")
        st.stop()

model = load_classification_model()
class_names = ["downdog", "goddess", "plank", "tree", "warrior2"]

# --- Initialiser MediaPipe Pose ---
mp_pose = mp.solutions.pose
@st.cache_resource
def get_pose_detector():
    return mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

pose_detector = get_pose_detector()

# --- Fonctions utilitaires ---
def extract_pose_landmarks(image: np.ndarray) -> np.ndarray:
    # Convertir BGR en RGB pour MediaPipe
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(rgb)
    if not results.pose_landmarks:
        return None
    lm = results.pose_landmarks.landmark
    return np.array([[p.x, p.y] for p in lm])


def compute_similarity_score(user_kp: np.ndarray, ref_kp: np.ndarray) -> float:
    if user_kp is None or ref_kp is None:
        return 0.0
    m = min(len(user_kp), len(ref_kp))
    dists = np.linalg.norm(user_kp[:m] - ref_kp[:m], axis=1)
    avg = np.mean(dists)
    score = max(0.0, 100.0 - avg * 1000.0)
    return round(score, 1)

# Dictionnaire des images de référence (URL ou chemins locaux)
ref_images = {
    "downdog": "https://.../Downdog-Ref.jpg",
    "goddess": "https://.../Goddess-Ref.jpg",
    "plank": "https://.../Plank-Ref.jpg",
    "tree": "https://.../Tree-Ref.jpg",
    "warrior2": "https://.../Warrior2-Ref.jpg"
}

# --- Classification de posture ---
def classify_pose(img_array: np.ndarray) -> tuple[str, float]:
    x = np.expand_dims(img_array, 0)
    preds = model.predict(x)
    idx = int(np.argmax(preds[0]))
    conf = float(preds[0][idx]) * 100.0
    return class_names[idx], conf

# --- Interface utilisateur ---
st.title("Analyse de Postures de Yoga 🧘‍♀️")
st.markdown("Téléchargez une photo de votre posture pour obtenir classification et score de similarité.")

col1, col2 = st.columns(2)
with col1:
    file = st.file_uploader("Choisissez une image (jpg/png)", type=["jpg", "jpeg", "png"])
    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, use_column_width=True, caption="Image téléchargée")
        arr = np.array(image.resize((224,224))) / 255.0
        bgr = cv2.cvtColor((arr*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if st.button("Analyser ma posture"):
            pose, conf = classify_pose(arr)
            user_kp = extract_pose_landmarks(bgr)
            ref_url = ref_images.get(pose)
            sim_score = None
            if ref_url:
                r = requests.get(ref_url, stream=True)
                if r.status_code == 200:
                    tmp = np.frombuffer(r.content, np.uint8)
                    ref = cv2.imdecode(tmp, cv2.IMREAD_COLOR)
                    ref_kp = extract_pose_landmarks(ref)
                    sim_score = compute_similarity_score(user_kp, ref_kp)
            st.session_state["pose"] = pose
            st.session_state["conf"] = conf
            st.session_state["sim_score"] = sim_score
            st.session_state["analyzed"] = True

with col2:
    if st.session_state.get("analyzed", False):
        st.subheader("Résultats")
        st.write(f"**Posture détectée:** {st.session_state['pose'].capitalize()} ({st.session_state['conf']:.1f}%)")
        if st.session_state['sim_score'] is not None:
            st.write(f"**Score de similarité:** {st.session_state['sim_score']}/100")
            st.progress(st.session_state['sim_score']/100)
        else:
            st.error("Impossible de calculer le score de similarité.")
        tips = {
            'downdog': ["- Alignez mains et épaules", "- Poussez hanches vers le haut"],
            'tree': ["- Regard fixe", "- Engagez le core"],
            'warrior2': ["- Genou aligné à la cheville", "- Bras parallèles"],
            'plank': ["- Corps en ligne droite", "- Engagez abdos"],
            'goddess': ["- Genoux sur chevilles", "- Engagez le core"]
        }
        for tip in tips.get(st.session_state['pose'], []):
            st.markdown(tip)
    else:
        st.info("Téléchargez une image et cliquez sur 'Analyser ma posture'.")

st.markdown("---")
st.markdown("_Application prête pour classification et évaluation des postures de yoga._")
