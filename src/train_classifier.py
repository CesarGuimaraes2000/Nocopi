import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path
import os 
from utils import verificar_e_baixar_nltk

if __name__ == "__main__":
    verificar_e_baixar_nltk()
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # --- ALTERAÇÃO 1: APONTAR PARA O NOVO CAMINHO DO CSV ---
    # O script agora procura o arquivo dentro da pasta 'generated_data'
    FEATURES_PATH = PROJECT_ROOT / 'generated_data' / 'features.csv'
    # ----------------------------------------------------

    # --- ALTERAÇÃO 2: DEFINIR UM DIRETÓRIO DE SAÍDA PARA O MODELO ---
    MODEL_DIR = PROJECT_ROOT / 'models'
    # Cria a pasta 'models' se ela não existir
    os.makedirs(MODEL_DIR, exist_ok=True)
    MODEL_OUTPUT_PATH = MODEL_DIR / 'plagiarism_classifier.joblib'
    # ----------------------------------------------------
    
    print(f"Carregando o dataset de features de '{FEATURES_PATH}'...")
    try:
        df = pd.read_csv(FEATURES_PATH)
    except FileNotFoundError:
        print(f"ERRO: Arquivo 'features.csv' não encontrado no caminho esperado.")
        print("Certifique-se de que você executou o script '1_generate_features.py' primeiro.")
        exit()

    # Separa as features (X) do rótulo (y)
    X = df[['similaridade_tfidf']]
    y = df['plagio']

    # Divide os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dados divididos: {len(X_train)} exemplos de treino, {len(X_test)} exemplos de teste.")

    # Cria e treina o modelo de Machine Learning
    print("Treinando o modelo de Regressão Logística...")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Avalia o modelo no conjunto de teste
    print("Avaliando o modelo...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo no conjunto de teste: {accuracy:.2%}")

    # Salva o modelo treinado no novo diretório
    print(f"Salvando o modelo em '{MODEL_OUTPUT_PATH}'...")
    joblib.dump(model, MODEL_OUTPUT_PATH)
    
    print("\nModelo treinado e salvo com sucesso!")