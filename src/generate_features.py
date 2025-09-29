import os
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from utils import verificar_e_baixar_nltk, calcular_similaridade_tfidf, preprocessar_texto

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATASET_PATH = PROJECT_ROOT / 'Datasets/Plagio' # Corrigido para seu caminho
    CSV_PATH = DATASET_PATH / 'file_information.csv'
    FILES_PATH = DATASET_PATH / 'data'
    verificar_e_baixar_nltk()
    print("Lendo o arquivo de informações do dataset...")
    df_info = pd.read_csv(CSV_PATH)
    
    dados_treino = []
    
    # Agrupa os arquivos por tarefa
    tasks = df_info['Task'].unique()
    
    print(f"Encontradas {len(tasks)} tarefas únicas. Processando para gerar pares...")
    for task in tasks:
        # Encontra o arquivo original para esta tarefa
        df_task = df_info[df_info['Task'] == task]
        orig_row = df_task[df_task['Category'] == 'orig']
        
        # Pula se por algum motivo não houver arquivo original para esta tarefa
        if orig_row.empty:
            continue
            
        orig_filename = orig_row['File'].iloc[0]
        with open(os.path.join(FILES_PATH, orig_filename), 'r', encoding='utf-8', errors='ignore') as f:
            texto_original = f.read()

        # Encontra todos os arquivos plagiados para esta tarefa
        plagiado_rows = df_task[df_task['Category'] != 'orig']
        
        for index, row in plagiado_rows.iterrows():
            plagiado_filename = row['File']
            with open(os.path.join(FILES_PATH, plagiado_filename), 'r', encoding='utf-8', errors='ignore') as f:
                texto_plagiado = f.read()

            # --- 1. Cria um PAR POSITIVO (plágio verdadeiro) ---
            # Compara o arquivo plagiado com seu verdadeiro original
            sim_tfidf_pos = calcular_similaridade_tfidf(texto_plagiado, texto_original)
            dados_treino.append([sim_tfidf_pos, 1])

            # --- 2. Cria um PAR NEGATIVO (plágio falso) ---
            # Compara o arquivo plagiado com um original de OUTRA tarefa
            outras_tasks = list(tasks)
            outras_tasks.remove(task)
            task_aleatoria = np.random.choice(outras_tasks)
            
            # Pega o nome do arquivo original da tarefa aleatória
            orig_aleatorio_filename = df_info[(df_info['Task'] == task_aleatoria) & (df_info['Category'] == 'orig')]['File'].iloc[0]
            with open(os.path.join(FILES_PATH, orig_aleatorio_filename), 'r', encoding='utf-8', errors='ignore') as f:
                texto_original_falso = f.read()

            sim_tfidf_neg = calcular_similaridade_tfidf(texto_plagiado, texto_original_falso)
            dados_treino.append([sim_tfidf_neg, 0])

    # Cria o DataFrame final a partir dos dados coletados
    df_features = pd.DataFrame(dados_treino, columns=['similaridade_tfidf', 'plagio'])
    
    # --- ALTERAÇÃO PARA ORGANIZAR A SAÍDA ---
    # 1. Define o caminho para a nova pasta de dados gerados
    output_dir = PROJECT_ROOT / 'generated_data'

    # 2. Cria a pasta, se ela não existir (o 'exist_ok=True' evita erros se a pasta já foi criada)
    os.makedirs(output_dir, exist_ok=True)

    # 3. Define o caminho completo para o arquivo CSV dentro da nova pasta
    output_path = output_dir / 'features.csv'
    # -----------------------------------------

    # Salva o arquivo no novo local
    df_features.to_csv(output_path, index=False)
    
    print(f"\nArquivo 'features.csv' criado com sucesso em '{output_path}'")
    print(f"Total de exemplos gerados: {len(df_features)}")
    print("Exemplo do dataset gerado:")
    print(df_features.head())
    print("\nVerificando a distribuição das classes:")
    print(df_features['plagio'].value_counts())