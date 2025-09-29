import streamlit as st
import joblib
import os
import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# --- Funções Auxiliares e de Processamento ---
# (Reunimos as funções essenciais dos outros scripts aqui)

def preprocessar_texto(texto):
    """Limpa e prepara o texto para análise."""
    stopwords = nltk.corpus.stopwords.words('english')
    texto = texto.lower()
    texto = re.sub(r'[^a-zA-Z\s]', repl='', string=texto, flags=re.I|re.A)
    tokens = nltk.word_tokenize(texto)
    tokens_filtrados = [palavra for palavra in tokens if palavra not in stopwords]
    return " ".join(tokens_filtrados)

def calcular_similaridade_tfidf(texto1, texto2):
    """Calcula a similaridade de cosseno TF-IDF entre dois textos."""
    vectorizer = TfidfVectorizer()
    textos_processados = [preprocessar_texto(texto1), preprocessar_texto(texto2)]
    if not textos_processados[0] or not textos_processados[1]:
        return 0.0
    matriz_tfidf = vectorizer.fit_transform(textos_processados)
    return cosine_similarity(matriz_tfidf[0:1], matriz_tfidf[1:2])[0][0]

# --- Funções de Carregamento com Cache ---
# O @st.cache_data garante que a base de dados seja carregada apenas uma vez,
# tornando o app muito mais rápido após a primeira execução.

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

# --- Interface Gráfica do Streamlit ---

# Título da Aplicação
st.title("🔎 Ferramenta de Detecção de Plágio")
st.markdown("Envie um arquivo `.txt` e escolha um modelo para analisar se ele é plágio de algum texto da nossa base de dados.")

# Definição dos caminhos
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / 'Datasets/Plagio'
CSV_PATH = DATASET_PATH / 'file_information.csv'
FILES_PATH = DATASET_PATH / 'data'
MODEL_PATH = PROJECT_ROOT / 'models' / 'plagiarism_classifier.joblib'

# Carrega os dados e o modelo usando as funções com cache
base_dados_originais = carregar_base_originais(CSV_PATH, FILES_PATH)
modelo_ml = carregar_modelo_ml(MODEL_PATH)

# Seletor para escolher o modelo de análise
modelo_escolhido = st.selectbox(
    "Escolha o modelo de análise:",
    ("Modelo 1: Similaridade Direta (TF-IDF)", "Modelo 2: Classificador com Machine Learning")
)

# Widget para upload de arquivo
uploaded_file = st.file_uploader("Envie seu arquivo de texto (.txt)", type=["txt"])

# Botão de análise e lógica principal
if uploaded_file is not None:
    # Lê o conteúdo do arquivo enviado
    texto_suspeito = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    
    with st.expander("Ver texto enviado"):
        st.text_area("", texto_suspeito, height=250)

    if st.button("Analisar Texto"):
        if not base_dados_originais:
            st.error("A base de dados de originais não pôde ser carregada. Verifique os caminhos no código.")
        else:
            with st.spinner("Analisando... Isso pode levar um momento."):
                plagio_encontrado = False
                
                # Itera sobre a base de dados
                for nome_original, texto_original in base_dados_originais.items():
                    similaridade = calcular_similaridade_tfidf(texto_suspeito, texto_original)
                    
                    # Lógica para o MODELO 1
                    if "Modelo 1" in modelo_escolhido:
                        # Usamos um limiar simples para o modelo 1
                        if similaridade > 0.85: # Limiar de 85%
                            plagio_encontrado = True
                            st.success("PLÁGIO DETECTADO!")
                            st.write(f"O texto parece ser plágio do arquivo de origem: **{nome_original}**")
                            st.write(f"Similaridade TF-IDF calculada: **{similaridade:.2%}**")
                            break
                    
                    # Lógica para o MODELO 2
                    elif "Modelo 2" in modelo_escolhido:
                        if not modelo_ml:
                            st.error("Modelo de Machine Learning não carregado.")
                            break
                        
                        dados_para_previsao = pd.DataFrame({'similaridade_tfidf': [similaridade]})
                        predicao = modelo_ml.predict(dados_para_previsao)[0]
                        
                        if predicao == 1:
                            plagio_encontrado = True
                            probabilidades = modelo_ml.predict_proba(dados_para_previsao)
                            confianca = probabilidades[0][1] # Confiança na classe "plágio"
                            st.success("PLÁGIO DETECTADO!")
                            st.write(f"O modelo de Machine Learning classificou este texto como plágio.")
                            st.write(f"Fonte mais provável: **{nome_original}**")
                            st.write(f"Similaridade TF-IDF com a fonte: **{similaridade:.2%}**")
                            st.write(f"Confiança da predição: **{confianca:.2%}**")
                            break
                
                if not plagio_encontrado:
                    st.info("Nenhuma correspondência de plágio encontrada na base de dados.")