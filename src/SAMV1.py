import os
import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# FUNÇÃO DE PRÉ-PROCESSAMENTO DE TEXTO PARA REMOÇÃO DE STOPWORDS E PONTUAÇÃO
def preprocessar_texto(texto):
    """Limpa e prepara o texto para análise."""
    stopwords = nltk.corpus.stopwords.words('english')
    texto = texto.lower()
    texto = re.sub(r'[^a-zA-Z\s]', repl='', string=texto, flags=re.I|re.A)
    tokens = nltk.word_tokenize(texto)
    tokens_filtrados = [palavra for palavra in tokens if palavra not in stopwords]
    return " ".join(tokens_filtrados)

# 1. DEFINIÇÃO DINÂMICA DOS CAMINHOS DOS ARQUIVOS
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / 'Datasets/Plagio'
CSV_PATH = DATASET_PATH / 'file_information.csv'
FILES_PATH = DATASET_PATH / 'data'

# 2. CARREGAMENTO DOS DOCUMENTOS USANDO O CSV 
print("Lendo o arquivo de informações do dataset...")
try:
    df_info = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"Erro: Arquivo '{CSV_PATH}' não encontrado. Verifique a estrutura das pastas.")
    exit()

df_originais = df_info[df_info['Category'] == 'orig']
documentos_referencia = []
nomes_referencia = []
print("Carregando documentos de referência (originais)...")
for index, row in df_originais.iterrows():
    filename = row['File']
    filepath = os.path.join(FILES_PATH, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        documentos_referencia.append(f.read())
        nomes_referencia.append(filename)

# 3. ESCOLHA DE UM ARQUIVO SUSPEITO PARA TESTE (não muda)
ARQUIVO_SUSPEITO = 'g0pD_taskc.txt'
categoria_real_suspeito = df_info[df_info['File'] == ARQUIVO_SUSPEITO]['Category'].iloc[0]

print(f"\nCarregando arquivo suspeito: '{ARQUIVO_SUSPEITO}' (Categoria real: {categoria_real_suspeito})")
with open(os.path.join(FILES_PATH, ARQUIVO_SUSPEITO), 'r', encoding='utf-8') as f:
    texto_suspeito = f.read()

# 4. PRÉ-PROCESSAMENTO, VETORIZAÇÃO E CÁLCULO (não muda)
print("Pré-processando todos os textos...")
referencia_processada = [preprocessar_texto(doc) for doc in documentos_referencia]
suspeito_processado = preprocessar_texto(texto_suspeito)
todos_os_textos = referencia_processada + [suspeito_processado]

print("Calculando a similaridade TF-IDF...")
vectorizer = TfidfVectorizer()
matriz_tfidf = vectorizer.fit_transform(todos_os_textos)
similaridades = cosine_similarity(matriz_tfidf[-1:], matriz_tfidf[:-1])
scores_similaridade = similaridades[0]

# 5. APRESENTAÇÃO DOS RESULTADOS (não muda)
print("\n--- Resultados da Análise de Plágio ---")
print(f"Arquivo Analisado: '{ARQUIVO_SUSPEITO}'")
print(f"Categoria Real Conhecida: '{categoria_real_suspeito}'")
print("-" * 40)
resultados = list(zip(nomes_referencia, scores_similaridade))
resultados_ordenados = sorted(resultados, key=lambda item: item[1], reverse=True)
print("Top 3 documentos originais mais similares:")
for nome_arquivo, score in resultados_ordenados[:3]:
    print(f"  -> Arquivo: {nome_arquivo} | Similaridade: {score:.2%}")

melhor_match = resultados_ordenados[0]
if melhor_match[1] > 0.80:
    print(f"\nConclusão: Forte suspeita de plágio do arquivo '{melhor_match[0]}'.")
else:
    print("\nConclusão: Nenhuma correspondência de plágio significativa encontrada.")


