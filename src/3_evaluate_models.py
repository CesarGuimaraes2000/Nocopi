# src/5_evaluate_models.py

import os
import pandas as pd
from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Importa as funções que precisamos
from utils import calcular_similaridade_tfidf
from data_loader import carregar_dados_parasci # Vamos adaptar para carregar o teste

def carregar_dados_teste_parasci(parasci_base_path):
    """Carrega o CONJUNTO DE TESTE do ParaSCI."""
    test_folder_path = parasci_base_path / 'ParaSCI-ACL' / 'test'
    src_file_path = test_folder_path / 'test.src'
    tgt_file_path = test_folder_path / 'test.tgt'
    
    test_pairs = []
    if os.path.exists(src_file_path) and os.path.exists(tgt_file_path):
        with open(src_file_path, 'r', encoding='utf-8') as f_src, \
             open(tgt_file_path, 'r', encoding='utf-8') as f_tgt:
            for sent_src, sent_tgt in zip(f_src, f_tgt):
                test_pairs.append((sent_src.strip(), sent_tgt.strip()))
    return test_pairs

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # --- 1. Carregar os modelos ---
    print("Carregando modelos para avaliação...")
    # Modelo 2 (Classificador ML)
    model_v2 = joblib.load(PROJECT_ROOT / 'models' / 'plagiarism_classifier.joblib')
    # Modelo 3 (Siamês Fine-tuned)
    model_v3_path = str(PROJECT_ROOT / 'models' / 'samv3-all-mpnet-base-v2-parasci')
    model_v3 = SentenceTransformer(model_v3_path)

    # --- 2. Carregar e preparar os dados de teste ---
    print("Carregando dados de teste do ParaSCI...")
    PARASCI_BASE_PATH = PROJECT_ROOT / 'Datasets' / 'ParaSCI' / 'Data'
    positive_pairs = carregar_dados_teste_parasci(PARASCI_BASE_PATH)
    
    # Cria pares negativos para um teste balanceado
    sents1 = [p[0] for p in positive_pairs]
    sents2 = [p[1] for p in positive_pairs]
    negative_pairs = list(zip(sents1, np.random.permutation(sents2)))

    test_data = positive_pairs + negative_pairs
    true_labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

    # --- 3. Avaliar cada modelo ---
    print(f"\nIniciando avaliação com {len(test_data)} pares de teste...")
    
    # Listas para guardar as predições de cada modelo
    preds_v1, preds_v2, preds_v3 = [], [], []
    
    # Limiares para os modelos de similaridade
    THRESHOLD_V1 = 0.85
    THRESHOLD_V3 = 0.80 # Modelos semânticos costumam ter scores mais altos, ajustamos o limiar

    for sent1, sent2 in test_data:
        # Avaliação do Modelo 1 (TF-IDF Direto)
        sim_tfidf = calcular_similaridade_tfidf(sent1, sent2)
        preds_v1.append(1 if sim_tfidf > THRESHOLD_V1 else 0)

        # Avaliação do Modelo 2 (Classificador ML)
        pred_v2 = model_v2.predict(pd.DataFrame({'similaridade_tfidf': [sim_tfidf]}))[0]
        preds_v2.append(pred_v2)

        # Avaliação do Modelo 3 (SAMV3 Siamês)
        embeddings = model_v3.encode([sent1, sent2])
        sim_sbert = util.cos_sim(embeddings[0], embeddings[1]).item()
        preds_v3.append(1 if sim_sbert > THRESHOLD_V3 else 0)

    # --- 4. Calcular e exibir as métricas ---
    def print_metrics(model_name, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        print(f"\n--- {model_name} ---")
        print(f"Acurácia: {accuracy:.2%}")
        print(f"Precisão: {precision:.2%}")
        print(f"Recall:   {recall:.2%}")
        print(f"F1-Score: {f1:.2%}")

    print_metrics("Modelo 1 (TF-IDF Direto)", true_labels, preds_v1)
    print_metrics("Modelo 2 (Classificador ML)", true_labels, preds_v2)
    print_metrics("Modelo 3 (SAMV3 Siamês)", true_labels, preds_v3)