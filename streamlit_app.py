import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Configuration de la page
st.set_page_config(
    page_title="Analyse de Postures de Yoga",
    page_icon="🧘‍♀️",
    layout="wide"
)

# Chargement du modèle avec cache
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenetv2_yoga_postures.h5")

model = load_model()

# Liste des postures correspondant à l’ordre de la sortie du modèle
POSE_CLASSES = ["downdog", "goddess", "plank", "tree", "warrior2"]

# Fonction de classification réelle
def classify_pose(img_array):
    img_input = np.expand_dims(img_array, axis=0)  # ajout d'une dimension batch
    predictions = model.predict(img_input)[0]  # prédictions
    predicted_idx = np.argmax(predictions)
    confidence = predictions[predicted_idx]
    return POSE_CLASSES[predicted_idx], confidence * 100

# Fonction de simulation pour le scoring (à remplacer par MediaPipe si besoin)
def compute_pose_score(img_array):
    return np.random.uniform(60, 95)

# Interface utilisateur
st.title("Analyse de Postures de Yoga")
st.markdown("Téléchargez une photo de votre posture pour obtenir une évaluation")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Téléchargez votre image")
    uploaded_file = st.file_uploader("Choisissez une image de posture de yoga", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Image téléchargée", use_column_width=True)
        
        # Prétraitement de l'image
        img_array = np.array(image.resize((224, 224))) / 255.0
        
        if st.button("Analyser ma posture"):
            with st.spinner("Analyse en cours..."):
                pose_name, confidence = classify_pose(img_array)
                score = compute_pose_score(img_array)
                
                st.session_state.pose_name = pose_name
                st.session_state.confidence = confidence
                st.session_state.score = score
                st.session_state.analyzed = True

with col2:
    st.subheader("Résultats de l'analyse")
    
    if 'analyzed' in st.session_state and st.session_state.analyzed:
        st.markdown(f"### Posture détectée: **{st.session_state.pose_name.capitalize()}**")
        st.markdown(f"Confiance: {st.session_state.confidence:.1f}%")
        
        st.markdown("### Score de qualité")
        score = st.session_state.score
        st.progress(score / 100)
        st.markdown(f"<h1 style='text-align: center; color: {'green' if score >= 80 else 'orange' if score >= 60 else 'red'};'>{score:.1f}/100</h1>", unsafe_allow_html=True)
        
        if score >= 80:
            st.success("Excellent! Votre posture est très bien exécutée.")
        elif score >= 60:
            st.warning("Bien! Votre posture est correcte mais peut être améliorée.")
        else:
            st.error("À améliorer. Essayez d'ajuster votre posture selon les principes du yoga.")
        
        st.subheader("Conseils d'amélioration")
        
        if st.session_state.pose_name == "downdog":
            st.markdown("""
            - Assurez-vous que vos mains sont à la largeur des épaules  
            - Poussez vos hanches vers le haut et l'arrière  
            - Gardez votre dos droit et vos talons près du sol  
            """)
        elif st.session_state.pose_name == "tree":
            st.markdown("""
            - Fixez votre regard sur un point fixe pour l'équilibre  
            - Gardez votre hanche ouverte et votre genou pointé vers l'extérieur  
            - Engagez votre core pour plus de stabilité  
            """)
        elif st.session_state.pose_name == "warrior2":
            st.markdown("""
            - Alignez votre genou avant avec votre cheville  
            - Gardez vos bras parallèles au sol  
            - Ouvrez votre poitrine et regardez au-dessus de votre main avant  
            """)
        elif st.session_state.pose_name == "plank":
            st.markdown("""
            - Gardez votre corps en ligne droite de la tête aux talons  
            - Engagez vos abdominaux et vos jambes  
            - Répartissez votre poids uniformément entre vos mains et vos orteils  
            """)
        elif st.session_state.pose_name == "goddess":
            st.markdown("""
            - Gardez vos genoux au-dessus de vos chevilles  
            - Tournez vos genoux vers l'extérieur dans la direction de vos orteils  
            - Engagez votre core et gardez votre dos droit  
            """)
    else:
        st.info("Téléchargez une image et cliquez sur 'Analyser ma posture' pour voir les résultats ici.")

st.markdown("---")
st.markdown("Application développée pour l'analyse et l'évaluation des postures de yoga")
