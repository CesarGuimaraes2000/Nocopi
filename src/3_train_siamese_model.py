# src/4_train_siamese_model.py

import logging
import json
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from pathlib import Path
# Importamos nossa função de carregamento de dados já ajustada
from data_loader import carregar_dados_parasci

# Configura o logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def iniciar_treinamento(config):
    """
    Função principal que orquestra o processo de fine-tuning.
    """
    logging.info(f"Carregando modelo base: {config['model_name']}")
    model = SentenceTransformer(config['model_name'])

    logging.info("Carregando dados de treinamento do ParaSCI...")
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    PARASCI_BASE_PATH = PROJECT_ROOT / 'Datasets' / 'ParaSCI' / 'Data'
    train_samples = carregar_dados_parasci(PARASCI_BASE_PATH)

    if not train_samples:
        logging.error("Nenhum dado de treinamento encontrado. Encerrando.")
        return

    # Configura o DataLoader padrão do PyTorch, que funciona bem com a OnlineContrastiveLoss
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=config['batch_size'])

    # Usa a OnlineContrastiveLoss, que funciona com os pares de paráfrases que carregamos
    train_loss = losses.OnlineContrastiveLoss(model=model)

    # O resto da configuração continua o mesmo
    warmup_steps = int(len(train_dataloader) * config['epochs'] * config['warmup_percentage'])
    output_path = str(PROJECT_ROOT / 'models' / config['output_model_name'])
    
    logging.info("\n--- Iniciando o Fine-tuning ---")
    logging.info(f"Dataset: ParaSCI-ACL ({len(train_samples)} exemplos)")
    logging.info(f"Função de Perda: OnlineContrastiveLoss")
    logging.info(f"Batch Size: {config['batch_size']}")
    logging.info(f"Épocas: {config['epochs']}")
    logging.info(f"Destino: {output_path}")

    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=config['epochs'],
              warmup_steps=warmup_steps,
              output_path=output_path,
              show_progress_bar=True)

    logging.info(f"\n--- Treinamento Concluído com Sucesso! ---")
    logging.info(f"Modelo SAMV3 salvo em: {output_path}")

# Ponto de entrada que lê o JSON e chama a função de treino
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    # Ajuste o nome do arquivo de configuração se você o nomeou de forma diferente
    CONFIG_FILE_PATH = PROJECT_ROOT / 'V3_training_config.json' 

    print(f"Carregando configurações de '{CONFIG_FILE_PATH}'...")
    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            training_config = json.load(f)
    except FileNotFoundError:
        print(f"ERRO: Arquivo de configuração '{CONFIG_FILE_PATH}' não encontrado.")
        exit()

    iniciar_treinamento(training_config)