# src/3_prepare_data.py

from pathlib import Path
from data_loader import carregar_dados_sts, carregar_dados_parasci

if __name__ == "__main__":
    print("--- Processando STS-B ---")
    train_samples_sts = carregar_dados_sts()
    if train_samples_sts:
        print("Exemplo de dado STS-B:")
        print(train_samples_sts[0])

    print("\n--- Processando ParaSCI ---")
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    PARASCI_BASE_PATH = PROJECT_ROOT / 'Datasets' / 'ParaSCI' / 'Data'
    train_samples_parasci = carregar_dados_parasci(PARASCI_BASE_PATH)
    if train_samples_parasci:
        print("Exemplo de dado ParaSCI:")
        print(train_samples_parasci[0])

    print("\nPreparação de dados concluída.")