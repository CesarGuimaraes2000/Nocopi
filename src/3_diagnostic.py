# src/5_evaluate_models.py (versão de diagnóstico)

import os
import pandas as pd
from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm

def carregar_dados_teste_parasci(parasci_base_path):
    # ... (esta função não muda)
    test_folder_path = parasci_base_path / 'ParaSCI-ACL' / 'test'
    src_file_path = test_folder_path / 'test.src'
    tgt_file_path = test_folder_path / 'test.tgt'
    test_pairs = []
    if os.path.exists(src_file_path) and os.path.exists(tgt_file_path):
        with open(src_file_path, 'r', encoding='utf-8') as f_src, open(tgt_file_path, 'r', encoding='utf-8') as f_tgt:
            for sent_src, sent_tgt in zip(f_src, f_tgt):
                test_pairs.append((sent_src.strip(), sent_tgt.strip()))
    return test_pairs

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # --- Carregar o modelo SAMV3 ---
    print("Carregando modelo SAMV3...")
    model_v3_path = str(PROJECT_ROOT / 'models' / 'samv3-all-mpnet-base-v2-parasci-BatchSize_16-Epochs_1-Warmup_0.1') # Use o nome da pasta do seu último treino
    model_v3 = SentenceTransformer(model_v3_path)

    # --- Carregar dados de teste ---
    print("Carregando dados de teste...")
    PARASCI_BASE_PATH = PROJECT_ROOT / 'Datasets' / 'ParaSCI' / 'Data'
    positive_pairs = carregar_dados_teste_parasci(PARASCI_BASE_PATH)
    sents1 = [p[0] for p in positive_pairs]
    sents2 = [p[1] for p in positive_pairs]
    negative_pairs = list(zip(sents1, np.random.permutation(sents2)))

    # --- SESSÃO DE DIAGNÓSTICO ---
    print("\n--- DIAGNÓSTICO DE SCORES DO SAMV3 ---")
    print("Analisando 5 pares POSITIVOS (paráfrases verdadeiras):")
    for s1, s2 in positive_pairs[:5]:
        embeddings = model_v3.encode([s1, s2], show_progress_bar=False)
        score = util.cos_sim(embeddings[0], embeddings[1]).item()
        print(f"  Score: {score:.4f}")

    print("\nAnalisando 5 pares NEGATIVOS (textos não relacionados):")
    for s1, s2 in negative_pairs[:5]:
        embeddings = model_v3.encode([s1, s2], show_progress_bar=False)
        score = util.cos_sim(embeddings[0], embeddings[1]).item()
        print(f"  Score: {score:.4f}")
    
    print("\n--- FIM DO DIAGNÓSTICO ---\n")
    # A partir daqui, você pode comentar o resto do script se quiser apenas ver o diagnóstico
    # input("Pressione Enter para continuar com a avaliação completa...")

    # --- Avaliação Completa (como antes) ---
    test_data = positive_pairs + negative_pairs
    true_labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)
    
    preds_v3 = []
    # Vamos testar com um limiar mais alto, por exemplo
    THRESHOLD_V3 = 0.95 
    print(f"Avaliando com um novo limiar de teste para o SAMV3: {THRESHOLD_V3}")

    for sent1, sent2 in tqdm(test_data, desc="Avaliando Modelo SAMV3"):
        embeddings = model_v3.encode([sent1, sent2], show_progress_bar=False)
        sim_sbert = util.cos_sim(embeddings[0], embeddings[1]).item()
        preds_v3.append(1 if sim_sbert > THRESHOLD_V3 else 0)

    def print_metrics(model_name, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        print(f"\n--- {model_name} ---")
        print(f"Acurácia: {accuracy:.2%}")
        print(f"Precisão: {precision:.2%}")
        print(f"Recall:   {recall:.2%}")
        print(f"F1-Score: {f1:.2%}")

    print_metrics("Modelo 3 (SAMV3 Siamês)", true_labels, preds_v3)