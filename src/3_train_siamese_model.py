import logging
import json
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from pathlib import Path
from data_loader import carregar_dados_sts

# Configura o logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def iniciar_treinamento_sts(config):
    """
    Função que orquestra o fine-tuning com o dataset STS-B e a CosineSimilarityLoss.
    """
    logging.info(f"Carregando modelo base: {config['model_name']}")
    model = SentenceTransformer(config['model_name'])

    logging.info("Carregando dados de treinamento do STS-B...")
    # A biblioteca baixa o arquivo 'stsbenchmark.tsv.gz' automaticamente se não o encontrar
    train_samples = carregar_dados_sts()
    
    if not train_samples:
        # Tenta uma segunda vez caso o download automático seja necessário
        from sentence_transformers import util
        sts_dataset_path = 'stsbenchmark.tsv.gz'
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)
        train_samples = carregar_dados_sts()
        if not train_samples:
            logging.error("Nenhum dado de treinamento do STS-B encontrado. Encerrando.")
            return

    # Configura o DataLoader para o treinamento
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=config['batch_size'])

    # Define a função de perda recomendada para dados com score graduado
    train_loss = losses.CosineSimilarityLoss(model=model)

    # O resto da configuração
    warmup_steps = int(len(train_dataloader) * config['epochs'] * 0.1) # Usa 10% de warmup
    output_path = str(Path(__file__).resolve().parent.parent / 'models' / config['output_model_name'])
    
    logging.info("\n--- Iniciando o Fine-tuning com STS-B ---")
    logging.info(f"Dataset: STS-B ({len(train_samples)} exemplos)")
    logging.info(f"Função de Perda: CosineSimilarityLoss")

    # Inicia o treinamento
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=config['epochs'],
              warmup_steps=warmup_steps,
              output_path=output_path,
              show_progress_bar=True,
              # Adiciona avaliação durante o treino para monitorar o progresso
              evaluation_steps=int(len(train_dataloader) * 0.1),
              output_path_save_best_model=True)

    logging.info(f"\n--- Treinamento Concluído! ---")
    logging.info(f"Modelo SAMV3 (versão STS-B) salvo em: {output_path}")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    CONFIG_FILE_PATH = PROJECT_ROOT / 'config.json'

    print(f"Carregando configurações de '{CONFIG_FILE_PATH}'...")
    with open(CONFIG_FILE_PATH, 'r') as f:
        training_config = json.load(f)

    iniciar_treinamento_sts(training_config)