import logging
import json
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from pathlib import Path
from data_loader import carregar_dados_parasci_contrastive # Importa nossa nova função

# Configura o logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def iniciar_treinamento_parasci(config):
    """
    Função que orquestra o fine-tuning com o dataset ParaSCI e a ContrastiveLoss.
    """
    logging.info(f"Carregando modelo base: {config['model_name']}")
    model = SentenceTransformer(config['model_name'])

    logging.info("Carregando dados de treinamento do ParaSCI...")
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    PARASCI_BASE_PATH = PROJECT_ROOT / 'Datasets' / 'ParaSCI' / 'Data'
    train_samples = carregar_dados_parasci_contrastive(PARASCI_BASE_PATH)

    if not train_samples:
        logging.error("Nenhum dado de treinamento do ParaSCI encontrado. Encerrando.")
        return

    # Configura o DataLoader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=config['batch_size'])

    # Define a função de perda Contrastive, ideal para pares com rótulo 0 ou 1
    train_loss = losses.ContrastiveLoss(model=model)

    # O resto da configuração
    warmup_steps = int(len(train_dataloader) * config['epochs'] * config['warmup_percentage'])
    output_path = str(PROJECT_ROOT / 'models' / config['output_model_name'])
    
    logging.info("\n--- Iniciando o Fine-tuning com ParaSCI ---")
    logging.info(f"Dataset: ParaSCI-ACL ({len(train_samples)} exemplos)")
    logging.info(f"Função de Perda: ContrastiveLoss")

    # Inicia o treinamento
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=config['epochs'],
              warmup_steps=warmup_steps,
              optimizer_params={'lr': config['learning_rate']},
              output_path=output_path,
              show_progress_bar=True)

    logging.info(f"\n--- Treinamento Concluído! ---")
    logging.info(f"Modelo SAMV3 (versão ParaSCI) salvo em: {output_path}")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    CONFIG_FILE_PATH = PROJECT_ROOT / 'V3_training_config.json' # Usa o novo arquivo de config

    print(f"Carregando configurações de '{CONFIG_FILE_PATH}'...")
    with open(CONFIG_FILE_PATH, 'r') as f:
        training_config = json.load(f)
    
    iniciar_treinamento_parasci(training_config)