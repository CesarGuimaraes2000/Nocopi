import os
import gzip
from pathlib import Path
from sentence_transformers.readers import InputExample

def carregar_dados_sts(sts_dataset_path='stsbenchmark.tsv.gz'):
    # (Esta função permanece a mesma, não precisa alterá-la)
    train_samples = []
    if not os.path.exists(sts_dataset_path):
        print(f"AVISO: Arquivo STS-B '{sts_dataset_path}' não encontrado.")
        return train_samples
    with gzip.open(sts_dataset_path, 'rt', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 7 and parts[0] == 'sts-train':
                score = float(parts[4]) / 5.0
                sent1 = parts[5]
                sent2 = parts[6]
                train_samples.append(InputExample(texts=[sent1, sent2], label=score))
    print(f"{len(train_samples)} exemplos carregados do STS-B (split de treino).")
    return train_samples

def carregar_dados_parasci(parasci_base_path):
    """
    Carrega o dataset ParaSCI a partir dos pares de arquivos .src e .tgt
    e retorna uma lista de InputExample com pares de paráfrases.
    """
    parasci_folder_path = parasci_base_path / 'ParaSCI-ACL' / 'train'
    src_file_path = parasci_folder_path / 'train.src'
    tgt_file_path = parasci_folder_path / 'train.tgt'

    train_samples = []
    if os.path.exists(src_file_path) and os.path.exists(tgt_file_path):
        print(f"Lendo pares de arquivos de '{parasci_folder_path}'...")
        with open(src_file_path, 'r', encoding='utf-8') as f_src, \
             open(tgt_file_path, 'r', encoding='utf-8') as f_tgt:
            
            for sent_src, sent_tgt in zip(f_src, f_tgt):
                sent1 = sent_src.strip()
                sent2 = sent_tgt.strip()
                # Cria o InputExample com o par de sentenças e o label 1 (paráfrase)
                train_samples.append(InputExample(texts=[sent1, sent2], label=1))

        print(f"{len(train_samples)} exemplos de paráfrases carregados do ParaSCI-ACL.")
    else:
        print(f"AVISO: Arquivos .src/.tgt do ParaSCI não encontrados em '{parasci_folder_path}'.")
    
    return train_samples