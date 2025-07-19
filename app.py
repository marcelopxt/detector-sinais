import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import PIL.Image
import json
import os
import gdown
import pandas as pd

# --- Caminhos dos Arquivos ---
MODEL_FILE_NAME = 'meu_modelo_gestos.keras'
CLASS_NAMES_FILE_NAME = 'class_names.json'

# --- IDs dos Arquivos no Google Drive ---
# ATENÇÃO: Substitua pelos IDs reais dos seus arquivos.
MODEL_DRIVE_ID = 'SEU_ID_DO_MODELO_AQUI'
CLASS_NAMES_DRIVE_ID = 'SEU_ID_DO_NOMES_CLASSES_AQUI'

# --- Funções de Carregamento e Pré-processamento ---

@st.cache_resource
def download_file_from_drive(file_id, output_path):
    """Baixa um arquivo do Google Drive se ele não existir localmente."""
    if not os.path.exists(output_path):
        st.write(f"Baixando {output_path} do Google Drive...")
        try:
            gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=False)
            st.success(f"Download de {output_path} concluído.")
        except Exception as e:
            st.error(f"Erro ao baixar {output_path}: {e}")
            st.stop()
    else:
        st.write(f"{output_path} já existe localmente.")

@st.cache_resource
def load_gesture_model(model_path):
    """Carrega o modelo de classificação de gestos."""
    download_file_from_drive(MODEL_DRIVE_ID, model_path)
    model = load_model(model_path, compile=False)
    return model

@st.cache_resource
def load_class_names(class_names_path):
    """Carrega os nomes das classes (gestos) a partir de um arquivo JSON."""
    download_file_from_drive(CLASS_NAMES_DRIVE_ID, class_names_path)
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    return class_names

def preprocess_image(image_data, target_height, target_width):
    """Pré-processa a imagem para o formato esperado pelo modelo."""
    img = PIL.Image.open(image_data).convert('RGB')
    img = img.resize((target_width, target_height))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Aplicação Principal Streamlit ---

def main():
    """Função principal que executa a aplicação Streamlit."""
    st.set_page_config(
        page_title="Classificador de Gestos de Linguagem de Sinais",
        page_icon="👋"
    )

    st.title('👋 Classificador de Gestos de Linguagem de Sinais')
    st.markdown("Faça o upload de uma imagem de um gesto para classificá-lo.")

    with st.spinner("Preparando o classificador... Isso pode levar alguns segundos na primeira vez."):
        model = load_gesture_model(MODEL_FILE_NAME)
        class_names = load_class_names(CLASS_NAMES_FILE_NAME)

    st.success("Classificador pronto! Por favor, faça o upload de uma imagem.")
    st.write("---")

    st.subheader('Faça o upload da imagem do gesto:')
    uploaded_file = st.file_uploader(
        "Arraste e solte uma imagem ou clique para selecionar",
        type=['png', 'jpg', 'jpeg']
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Imagem Carregada.', use_container_width=True)

        # Pré-processa a imagem e faz a predição
        processed_image = preprocess_image(uploaded_file, 50, 50)

        with st.spinner('Analisando o gesto...'):
            predictions = model.predict(processed_image)
            predicted_class_index = np.argmax(predictions)
            confidence = predictions[0][predicted_class_index]
            predicted_class_name = class_names[predicted_class_index]

        # Exibe o resultado principal
        st.subheader('Resultado da Classificação')
        st.metric(label="Gesto Previsto", value=predicted_class_name)
        st.metric(label="Confiança do Modelo", value=f"{confidence*100:.2f}%")

        # Animação divertida para predições com alta confiança
        if confidence > 0.8:
            st.balloons()

        # Exibe a tabela com as probabilidades de todas as classes
        st.write("---")
        st.subheader("Probabilidades para cada classe")
        df_probs = pd.DataFrame({
            'Gesto': class_names,
            'Probabilidade (%)': predictions[0] * 100
        })
        df_probs = df_probs.sort_values(by='Probabilidade (%)', ascending=False)
        st.dataframe(df_probs.set_index('Gesto'), use_container_width=True)

    st.markdown("---")
    st.info("Este aplicativo utiliza um modelo de Deep Learning para classificar gestos da linguagem de sinais a partir de imagens.")


if __name__ == "__main__":
    main()
