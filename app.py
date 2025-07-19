import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import PIL.Image
import json
import os
import gdown
import pandas as pd
# --- Caminhos dos Arquivos Salvos (ajuste seus IDs do Drive aqui se usar gdown) ---
MODEL_FILE_NAME = 'meu_modelo_gestos.keras'
CLASS_NAMES_FILE_NAME = 'class_names.json'

# --- **SUBSTITUA PELOS IDs REAIS DOS SEUS ARQUIVOS NO GOOGLE DRIVE** ---
MODEL_DRIVE_ID = 'SEU_ID_DO_MODELO_AQUI' # <--- OBRIGATÓRIO ATUALIZAR
CLASS_NAMES_DRIVE_ID = 'SEU_ID_DO_NOMES_CLASSES_AQUI' # <--- OBRIGATÓRIO ATUALIZAR

# Funções de carregamento (mantidas iguais)
@st.cache_resource
def download_file_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        st.write(f"Baixando {output_path} do Google Drive...")
        try:
            gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=False)
            st.write(f"Download de {output_path} concluído.")
        except Exception as e:
            st.error(f"Erro ao baixar {output_path} do Google Drive: {e}")
            st.stop()
    else:
        st.write(f"{output_path} já existe localmente. Pulando download.")

@st.cache_resource
def load_gesture_model(model_path):
    download_file_from_drive(MODEL_DRIVE_ID, model_path)
    model = load_model(model_path, compile=False)
    return model

@st.cache_resource
def load_class_names(class_names_path):
    download_file_from_drive(CLASS_NAMES_DRIVE_ID, class_names_path)
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    return class_names

# Função de Pré-processamento de Imagem para Previsão (mantida igual)
def preprocess_image(image_data, target_height, target_width):
    img = PIL.Image.open(image_data).convert('RGB')
    img = img.resize((target_width, target_height))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Função Principal do Aplicativo Streamlit ---
def main():
    st.set_page_config(page_title="Classificador de Gestos de Linguagem de Sinais")

    st.title('👋 Classificador de Gestos de Linguagem de Sinais')
    st.markdown("Faça o upload de uma imagem de um gesto de linguagem de sinais para classificá-lo.")

    st.info("Preparando o classificador... Isso pode levar alguns segundos na primeira vez (baixando o modelo).")
    
    model = load_gesture_model(MODEL_FILE_NAME)
    class_names = load_class_names(CLASS_NAMES_FILE_NAME)
    
    st.success("Classificador pronto! Faça o upload de uma imagem.")

    st.write("---")
    st.subheader('Faça o upload da imagem do gesto:')
    uploaded_file = st.file_uploader("Arraste e solte uma imagem ou clique para selecionar", type=['png','jpg','jpeg'])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Imagem Carregada.', use_container_width=True)
        st.success('Imagem carregada com sucesso!')

        processed_image = preprocess_image(uploaded_file, 50, 50)
        
        with st.spinner('Analisando gesto...'):
            predictions = model.predict(processed_image)
            
            predicted_class_index = np.argmax(predictions)
            confidence = predictions[0][predicted_class_index]
            
            predicted_class_name = class_names[predicted_class_index]

        st.subheader('Resultado da Classificação:')
        st.write(f"O modelo prevê que o gesto é: **{predicted_class_name}**")
        st.write(f"Confiança do modelo: **{confidence*100:.2f}%**")

        # --- Mova o bloco do confianca > 0.8 para cá, se quiser a animação condicional ---
        if confidence > 0.8:
            st.balloons()
        # --- Fim do bloco de animação condicional ---
        
        # --- Mova o cálculo e exibição de df_probs para fora de qualquer if de confiança ---
        # Ele deve sempre ser exibido, independentemente da confiança.
        st.write("---")
        st.subheader("Probabilidades para cada classe:")
        df_probs = pd.DataFrame({
            'Gesto': class_names,
            'Probabilidade (%)': predictions[0] * 100
        })
        df_probs = df_probs.sort_values(by='Probabilidade (%)', ascending=False)
        st.dataframe(df_probs.set_index('Gesto'))
        # --- Fim da exibição de df_probs ---

    st.markdown("---")
    st.write("Desenvolvido com TensorFlow e Streamlit")
    st.write("Para treinar o modelo, execute `python model_trainer.py`, faça o upload dos arquivos `.keras` e `.json` para o Google Drive e atualize os IDs neste script.")

if __name__ == "__main__":
    main()
