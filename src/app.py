import streamlit as st
import joblib
import os
import pandas as pd
from pathlib import Path

# Importando nossas novas funções de análise e as utilidades
from utils import verificar_e_baixar_nltk
from SAMV1 import analisar_com_tfidf
from SAMV2 import analisar_com_ml

# Chama a função de verificação no início da execução do app
verificar_e_baixar_nltk()

# --- Funções de Carregamento com Cache (continuam aqui) ---
@st.cache_data
def carregar_base_originais(csv_path, files_path):
    """Carrega todos os textos originais da base de dados e os armazena em cache."""
    try:
        df_info = pd.read_csv(csv_path)
        df_originais = df_info[df_info['Category'] == 'orig']
        base_dados_originais = {}
        for index, row in df_originais.iterrows():
            filename = row['File']
            filepath = os.path.join(files_path, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                base_dados_originais[filename] = f.read()
        return base_dados_originais
    except FileNotFoundError:
        st.error(f"ERRO: Dataset não encontrado nos caminhos especificados. Verifique as pastas.")
        return None

@st.cache_resource
def carregar_modelo_ml(model_path):
    """Carrega o modelo de Machine Learning e o armazena em cache."""
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"ERRO: Modelo '{model_path}' não encontrado. Execute o script '2_train_classifier.py' primeiro.")
        return None

# --- Interface Gráfica do Streamlit (agora muito mais limpa) ---

st.title("🔎 Ferramenta de Detecção de Plágio")
st.markdown("Envie um arquivo `.txt` e escolha um modelo para analisar se ele é plágio de algum texto da nossa base de dados.")

# Definição dos caminhos
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / 'Datasets/Plagio'
CSV_PATH = DATASET_PATH / 'file_information.csv'
FILES_PATH = DATASET_PATH / 'data'
MODEL_PATH = PROJECT_ROOT / 'models' / 'plagiarism_classifier.joblib'

# Carrega os dados e o modelo
base_dados_originais = carregar_base_originais(CSV_PATH, FILES_PATH)
modelo_ml = carregar_modelo_ml(MODEL_PATH)

# Seletor de modelo e upload de arquivo
modelo_escolhido = st.selectbox(
    "Escolha o modelo de análise:",
    ("Modelo 1: Similaridade Direta (TF-IDF)", "Modelo 2: Classificador com Machine Learning")
)
uploaded_file = st.file_uploader("Envie seu arquivo de texto (.txt)", type=["txt"])

if uploaded_file is not None:
    texto_suspeito = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    
    with st.expander("Ver texto enviado"):
        st.text_area("", texto_suspeito, height=250)

    if st.button("Analisar Texto"):
        if not base_dados_originais:
            st.error("A base de dados de originais não pôde ser carregada.")
        else:
            with st.spinner("Analisando..."):
                resultado = None
                if "Modelo 1" in modelo_escolhido:
                    resultado = analisar_com_tfidf(texto_suspeito, base_dados_originais, limiar=0.75)
                elif "Modelo 2" in modelo_escolhido:
                    resultado = analisar_com_ml(texto_suspeito, base_dados_originais, modelo_ml)
                
                # Bloco único para exibir os resultados
                if resultado and not resultado.get("erro"):
                    st.markdown("---") # Linha divisória
                    
                    # Exibe o veredito
                    if resultado.get("plagio_detectado"):
                        st.success("PLÁGIO DETECTADO!")
                    else:
                        st.info("Nenhuma correspondência de plágio encontrada.")

                    st.markdown("---")
                    
                    # Exibe os detalhes da melhor correspondência, independentemente do veredito
                    st.write(f"**Fonte mais próxima na base de dados:** `{resultado.get('fonte_provavel')}`")
                    st.write(f"**Similaridade TF-IDF com esta fonte:** `{resultado.get('similaridade'):.2%}`")
                    
                    # Se o Modelo 2 foi usado, exibe também a confiança
                    if 'confianca' in resultado and resultado.get('confianca') > 0:
                        st.write(f"**Confiança da predição (Modelo ML):** `{resultado.get('confianca'):.2%}`")

                elif resultado and resultado.get("erro"):
                    st.error(resultado.get("erro"))