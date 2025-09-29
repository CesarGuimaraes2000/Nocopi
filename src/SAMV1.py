import os
import pandas as pd
from pathlib import Path

from utils import verificar_e_baixar_nltk, calcular_similaridade_tfidf

verificar_e_baixar_nltk()

if __name__ == "__main__":
    # 1. DEFINIÇÃO DOS CAMINHOS
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATASET_PATH = PROJECT_ROOT / 'Datasets/Plagio'
    CSV_PATH = DATASET_PATH / 'file_information.csv'
    FILES_PATH = DATASET_PATH / 'data'

    # 2. CARREGAMENTO DOS DOCUMENTOS
    print("Lendo o arquivo de informações do dataset...")
    try:
        df_info = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{CSV_PATH}' não encontrado. Verifique a estrutura das pastas.")
        exit()

    df_originais = df_info[df_info['Category'] == 'orig']
    
    # Carrega os textos originais em um dicionário para fácil acesso
    base_dados_originais = {}
    print("Carregando documentos de referência (originais)...")
    for index, row in df_originais.iterrows():
        filename = row['File']
        filepath = os.path.join(FILES_PATH, filename)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            base_dados_originais[filename] = f.read()

    # 3. ESCOLHA DE UM ARQUIVO SUSPEITO
    ARQUIVO_SUSPEITO = 'g0pD_taskc.txt'
    
    df_suspeito = df_info[df_info['File'] == ARQUIVO_SUSPEITO]
    if df_suspeito.empty:
        print(f"ERRO: Arquivo suspeito '{ARQUIVO_SUSPEITO}' não encontrado no CSV.")
        exit()
    
    categoria_real_suspeito = df_suspeito['Category'].iloc[0]

    print(f"\nCarregando arquivo suspeito: '{ARQUIVO_SUSPEITO}' (Categoria real: {categoria_real_suspeito})")
    with open(os.path.join(FILES_PATH, ARQUIVO_SUSPEITO), 'r', encoding='utf-8', errors='ignore') as f:
        texto_suspeito = f.read()

    # 4. CÁLCULO DA SIMILARIDADE USANDO A FUNÇÃO DO UTILS (MUDANÇA PRINCIPAL)
    print("\nCalculando similaridade com a base de dados...")
    
    resultados_similaridade = []
    # Itera sobre cada documento original e calcula a similaridade individualmente
    for nome_original, texto_original in base_dados_originais.items():
        score = calcular_similaridade_tfidf(texto_suspeito, texto_original)
        resultados_similaridade.append((nome_original, score))

    # 5. APRESENTAÇÃO DOS RESULTADOS (lógica similar, mas com a nova lista de resultados)
    print("\n--- Resultados da Análise de Plágio (Similaridade Direta TF-IDF) ---")
    print(f"Arquivo Analisado: '{ARQUIVO_SUSPEITO}'")
    
    resultados_ordenados = sorted(resultados_similaridade, key=lambda item: item[1], reverse=True)
    
    print("\nTop 3 documentos originais mais similares:")
    for nome_arquivo, score in resultados_ordenados[:3]:
        print(f"  -> Arquivo: {nome_arquivo} | Similaridade: {score:.2%}")

    melhor_match = resultados_ordenados[0]
    if melhor_match[1] > 0.70:
        print(f"\nConclusão: Forte suspeita de plágio do arquivo '{melhor_match[0]}'.")
    else:
        print("\nConclusão: Nenhuma correspondência de plágio significativa encontrada.")