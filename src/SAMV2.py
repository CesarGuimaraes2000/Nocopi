import joblib
import os
import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from utils import verificar_e_baixar_nltk, calcular_similaridade_tfidf, preprocessar_texto

if __name__ == "__main__":
    verificar_e_baixar_nltk()
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MODEL_PATH = PROJECT_ROOT / 'models' / 'plagiarism_classifier.joblib'
    DATASET_PATH = PROJECT_ROOT / 'Datasets/Plagio'
    CSV_PATH = DATASET_PATH / 'file_information.csv'
    FILES_PATH = DATASET_PATH / 'data'
    
    # 1. Carrega o modelo treinado
    print("Carregando o modelo de classificação de plágio...")
    try:
        modelo_carregado = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"ERRO: Modelo não encontrado em '{MODEL_PATH}'. Execute o '2_train_classifier.py' primeiro.")
        exit()

    # 2. Carrega a base de dados de textos ORIGINAIS
    print("Carregando a base de dados de textos originais...")
    try:
        df_info = pd.read_csv(CSV_PATH)
        df_originais = df_info[df_info['Category'] == 'orig']
        
        # Cria um dicionário com {nome_do_arquivo: conteúdo} para fácil acesso
        base_dados_originais = {}
        for index, row in df_originais.iterrows():
            filename = row['File']
            filepath = os.path.join(FILES_PATH, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                base_dados_originais[filename] = f.read()
        print(f"{len(base_dados_originais)} textos originais carregados.")
    except FileNotFoundError:
        print(f"ERRO: Dataset não encontrado em '{DATASET_PATH}'. Verifique os caminhos.")
        exit()
        
    # 3. Pede ao usuário o caminho do arquivo a ser analisado
    print("\n--- Verificação de Plágio na Base de Dados ---")
    caminho_suspeito = input("Por favor, insira o caminho para o arquivo suspeito: ")

    try:
        with open(caminho_suspeito, 'r', encoding='utf-8', errors='ignore') as f:
            texto_suspeito = f.read()
    except FileNotFoundError:
        print(f"ERRO: Arquivo suspeito '{caminho_suspeito}' não encontrado.")
        exit()

    # 4. Itera sobre a base de dados e faz a predição para cada par
    print("\nAnalisando...")
    plagio_encontrado = False
    for nome_original, texto_original in base_dados_originais.items():
        # Calcula a feature de similaridade para o par atual
        similaridade = calcular_similaridade_tfidf(texto_suspeito, texto_original)
        
        # Prepara os dados para o modelo
        dados_para_previsao = pd.DataFrame({'similaridade_tfidf': [similaridade]})
        
        # Faz a predição
        predicao = modelo_carregado.predict(dados_para_previsao)[0]
        
        # Se o modelo prever "1" (plágio), encontramos uma correspondência
        if predicao == 1:
            plagio_encontrado = True
            print("\n--- Resultado da Análise ---")
            print(f"VEREDITO: PLÁGIO DETECTADO!")
            print(f"O texto parece ser plágio do arquivo de origem: '{nome_original}'")
            print(f"Similaridade TF-IDF calculada com a fonte: {similaridade:.2%}")
            break # Para a busca assim que encontrar o primeiro caso de plágio

    # 5. Se o loop terminar sem encontrar plágio
    if not plagio_encontrado:
        print("\n--- Resultado da Análise ---")
        print("VEREDITO: Nenhuma correspondência de plágio encontrada na base de dados.")