import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import mediapipe as mp

# Configuration de la page
st.set_page_config(
    page_title="Analyse de Postures de Yoga",
    page_icon="🧘‍♀️",
    layout="wide"
)

# Charger le modèle MobileNetV2 personnalisé
@st.cache_resource
def load_classification_model():
    # Assurez-vous d'avoir placé 'yoga_mobilenetv2.h5' à la racine du repo
    return tf.keras.models.load_model("yoga_mobilenetv2.h5")

model = load_classification_model()
class_names = ["downdog", "goddess", "plank", "tree", "warrior2"]

# Initialiser MediaPipe Pose
mp_pose = mp.solutions.pose
@st.cache_resource
def create_pose_detector():
    return mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

pose_detector = create_pose_detector()

# Fonctions métier

def classify_pose(img_array):
    # Préparer le batch
    x = np.expand_dims(img_array, axis=0)
    preds = model.predict(x)
    idx = np.argmax(preds[0])
    confidence = float(preds[0][idx]) * 100
    return class_names[idx], confidence


def compute_pose_score(img_array):
    # Convertir en BGR et uint8
    image_bgr = (img_array * 255).astype(np.uint8)[..., ::-1]
    results = pose_detector.process(image_bgr)
    if not results.pose_landmarks:
        return 0.0
    lm = results.pose_landmarks.landmark

    def angle(a, b, c):
        va = np.array([a.x - b.x, a.y - b.y])
        vc = np.array([c.x - b.x, c.y - b.y])
        cosang = np.dot(va, vc) / (np.linalg.norm(va) * np.linalg.norm(vc))
        return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

    # Exemple pour planche: angle épaule-hanche-genou gauche
    left_angle = angle(
        lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
        lm[mp_pose.PoseLandmark.LEFT_HIP],
        lm[mp_pose.PoseLandmark.LEFT_KNEE]
    )
    target = 180
    score = max(0, 100 - abs(left_angle - target))
    return score

# Interface utilisateur
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Téléchargez votre image")
    uploaded_file = st.file_uploader("Choisissez une image de posture de yoga", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image téléchargée", use_column_width=True)
        img_array = np.array(image.resize((224, 224))) / 255.0
        if st.button("Analyser ma posture"):
            with st.spinner("Analyse en cours..."):
                pose_name, confidence = classify_pose(img_array)
                score = compute_pose_score(img_array)
                st.session_state.update({
                    "pose_name": pose_name,
                    "confidence": confidence,
                    "score": score,
                    "analyzed": True
                })

with col2:
    st.subheader("Résultats de l'analyse")
    if st.session_state.get("analyzed", False):
        st.markdown(f"### Posture détectée: **{st.session_state['pose_name'].capitalize()}**")
        st.markdown(f"Confiance: {st.session_state['confidence']:.1f}%")
        st.markdown("### Score de qualité")
        prog = st.progress(st.session_state['score']/100)
        color = 'green' if st.session_state['score'] >= 80 else 'orange' if st.session_state['score'] >= 60 else 'red'
        st.markdown(f"<h1 style='text-align: center; color: {color};'>{st.session_state['score']:.1f}/100</h1>", unsafe_allow_html=True)
        if st.session_state['score'] >= 80:
            st.success("Excellent! Votre posture est très bien exécutée.")
        elif st.session_state['score'] >= 60:
            st.warning("Bien! Votre posture est correcte mais peut être améliorée.")
        else:
            st.error("À améliorer. Essayez d'ajuster votre posture selon les principes du yoga.")
        # Conseils
        mapping = {
            'downdog': [
                "- Assurez-vous que vos mains sont à la largeur des épaules",
                "- Poussez vos hanches vers le haut et l'arrière",
                "- Gardez votre dos droit et vos talons près du sol"
            ],
            'tree': [
                "- Fixez votre regard sur un point fixe pour l'équilibre",
                "- Gardez votre hanche ouverte et votre genou pointé vers l'extérieur",
                "- Engagez votre core pour plus de stabilité"
            ],
            'warrior2': [
                "- Alignez votre genou avant avec votre cheville",
                "- Gardez vos bras parallèles au sol",
                "- Ouvrez votre poitrine et regardez au-dessus de votre main avant"
            ],
            'plank': [
                "- Gardez votre corps en ligne droite de la tête aux talons",
                "- Engagez vos abdominaux et vos jambes",
                "- Répartissez votre poids uniformément entre vos mains et vos orteils"
            ],
            'goddess': [
                "- Gardez vos genoux au-dessus de vos chevilles",
                "- Tournez vos genoux vers l'extérieur dans la direction de vos orteils",
                "- Engagez votre core et gardez votre dos droit"
            ]
        }
        for line in mapping.get(st.session_state['pose_name'], []):
            st.markdown(line)
    else:
        st.info("Téléchargez une image et cliquez sur 'Analyser ma posture' pour voir les résultats ici.")

# Pied de page
st.markdown("---")
st.markdown("Application développée pour l'analyse et l'évaluation des postures de yoga")
