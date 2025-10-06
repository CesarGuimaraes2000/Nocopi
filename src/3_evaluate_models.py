import os
import pandas as pd
from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm

# Importa as funções que precisamos
from utils import calcular_similaridade_tfidf

def carregar_dados_teste_parasci(parasci_base_path):
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

    # --- 1. Carregar os modelos --- (não muda)
    print("Carregando modelos para avaliação...")
    model_v2 = joblib.load(PROJECT_ROOT / 'models' / 'plagiarism_classifier.joblib')
    model_v3_path = str(PROJECT_ROOT / 'models' / 'samv3-parasci-finetuned-BatchSize_48-Epochs_5-Warmup_0.1-learning_rate_2e-5')
    model_v3 = SentenceTransformer(model_v3_path)

    # --- 2. Carregar e preparar os dados de teste --- (não muda)
    print("Carregando dados de teste do ParaSCI...")
    PARASCI_BASE_PATH = PROJECT_ROOT / 'Datasets' / 'ParaSCI' / 'Data'
    positive_pairs = carregar_dados_teste_parasci(PARASCI_BASE_PATH)
    
    sents1 = [p[0] for p in positive_pairs]
    sents2 = [p[1] for p in positive_pairs]
    negative_pairs = list(zip(sents1, np.random.permutation(sents2)))

    test_data = positive_pairs + negative_pairs
    true_labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

    # --- 3. Avaliar cada modelo ---
    print(f"\nIniciando avaliação com {len(test_data)} pares de teste...")
    
    preds_v1, preds_v2, preds_v3 = [], [], []
    THRESHOLD_V1 = 0.85
    THRESHOLD_V3 = 0.80

    # <-- 2. "Envolvemos" o nosso loop com tqdm() -->
    # O 'desc' é o texto que aparecerá antes da barra de progresso.
    for sent1, sent2 in tqdm(test_data, desc="Avaliando Modelos"):
        # O corpo do loop não muda em nada
        sim_tfidf = calcular_similaridade_tfidf(sent1, sent2)
        preds_v1.append(1 if sim_tfidf > THRESHOLD_V1 else 0)

        pred_v2 = model_v2.predict(pd.DataFrame({'similaridade_tfidf': [sim_tfidf]}))[0]
        preds_v2.append(pred_v2)

        embeddings = model_v3.encode([sent1, sent2], show_progress_bar=False) # Desativamos a barra interna do SBERT
        sim_sbert = util.cos_sim(embeddings[0], embeddings[1]).item()
        preds_v3.append(1 if sim_sbert > THRESHOLD_V3 else 0)

    # --- 4. Calcular e exibir as métricas --- (não muda)
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