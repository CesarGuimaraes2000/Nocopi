import streamlit as st
import joblib
import os
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer 

# Importando nossas funções de análise e as utilidades
from utils import verificar_e_baixar_nltk
from SAMV1 import analisar_com_tfidf
from SAMV2 import analisar_com_ml
from SAMV3 import analisar_com_samv3 

# Chama a função de verificação no início da execução do app
verificar_e_baixar_nltk()

# --- Funções de Carregamento com Cache ---
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
    """Carrega o modelo de Machine Learning (SAMV2)."""
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"ERRO: Modelo SAMV2 '{model_path}' não encontrado. Execute o '2_train_classifier.py'.")
        return None

@st.cache_resource
def carregar_modelo_samv3(model_path):
    """Carrega o modelo Sentence Transformer (SAMV3)."""
    if not os.path.exists(model_path):
        st.error(f"ERRO: Modelo SAMV3 não encontrado em '{model_path}'. Execute o script de treinamento do SAMV3.")
        return None
    try:
        return SentenceTransformer(model_path)
    except Exception as e:
        st.error(f"ERRO ao carregar modelo SAMV3: {e}")
        return None
# -----------------------------------------


# --- Interface Gráfica do Streamlit ---

st.title("🔎 Ferramenta de Detecção de Plágio")
st.markdown("Envie um arquivo `.txt` e escolha um modelo para analisar se ele é plágio de algum texto da nossa base de dados.")

# Definição dos caminhos
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / 'Datasets/Plagio'
CSV_PATH = DATASET_PATH / 'file_information.csv'
FILES_PATH = DATASET_PATH / 'data'
MODEL_V2_PATH = PROJECT_ROOT / 'models' / 'plagiarism_classifier.joblib'
MODEL_V3_PATH = PROJECT_ROOT / 'models' / 'samv3-parasci-finetuned-BatchSize_48-Epochs_5-Warmup_0.1-learning_rate_2e-5' 

# Carrega todos os recursos necessários
base_dados_originais = carregar_base_originais(CSV_PATH, FILES_PATH)
modelo_v2 = carregar_modelo_ml(MODEL_V2_PATH)
modelo_v3 = carregar_modelo_samv3(str(MODEL_V3_PATH))


# Seletor de modelo e upload de arquivo
modelo_escolhido = st.selectbox(
    "Escolha o modelo de análise:",
    (
        "Modelo 1: Similaridade Direta (SAMV1)",
        "Modelo 2: Classificador com Machine Learning (SAMV2)",
        "Modelo 3: Semântico (SAMV3)",
    )
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
                    resultado = analisar_com_tfidf(texto_suspeito, base_dados_originais, limiar=0.85)
                elif "Modelo 2" in modelo_escolhido:
                    resultado = analisar_com_ml(texto_suspeito, base_dados_originais, modelo_v2)
                # --- NOVA LÓGICA PARA O MODELO 3 ---
                elif "Modelo 3" in modelo_escolhido:
                    resultado = analisar_com_samv3(texto_suspeito, base_dados_originais, modelo_v3, top_k=10, limiar_final=0.80)

                # Bloco único para exibir os resultados
                if resultado and not resultado.get("erro"):
                    st.markdown("---")
                    
                    if resultado.get("plagio_detectado"):
                        st.success("PLÁGIO DETECTADO!")
                    else:
                        st.info("Nenhuma correspondência de plágio encontrada.")

                    st.markdown("---")
                    st.write(f"**Fonte mais próxima na base de dados:** `{resultado.get('fonte_provavel')}`")
                    
                    # Exibe os scores de similaridade relevantes
                    if 'similaridade_semantica' in resultado:
                        st.write(f"**Similaridade Semântica (SAMV3):** `{resultado.get('similaridade_semantica'):.2%}`")
                        st.write(f"**Similaridade Lexical (TF-IDF) com esta fonte:** `{resultado.get('similaridade_tfidf'):.2%}`")
                    elif 'similaridade' in resultado:
                        st.write(f"**Similaridade TF-IDF com esta fonte:** `{resultado.get('similaridade'):.2%}`")
                    
                    if 'confianca' in resultado and resultado.get('confianca') > 0:
                        st.write(f"**Confiança da predição (Modelo ML):** `{resultado.get('confianca'):.2%}`")

                elif resultado and resultado.get("erro"):
                    st.error(resultado.get("erro"))