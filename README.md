# Detector de Plágio com Inteligência Artificial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CesarGuimaraes2000/Nocopi/blob/main/start_colab.ipynb)

Este projeto foi desenvolvido como parte da disciplina de Inteligência Artificial. O objetivo é explorar, implementar e comparar **três abordagens distintas** para a detecção de plágio em textos, utilizando datasets públicos para treinar e validar os modelos.

O projeto culmina em uma aplicação web interativa construída com Streamlit, que permite ao usuário testar e comparar os modelos em tempo real.

## Metodologia

O projeto implementa três metodologias principais, representando diferentes eras e níveis de complexidade da Inteligência Artificial:

1.  **IA Clássica (Recuperação de Informação):** Uma abordagem determinística baseada em estatística de texto.
2.  **Machine Learning Supervisionado (Classificação):** Uma abordagem onde um modelo simples "aprende" uma regra de decisão a partir de exemplos rotulados.
3.  **Deep Learning (Aprendizado de Métrica):** Uma abordagem de ponta que utiliza Redes Neurais Siamesas para aprender o *significado semântico* do texto, em vez de apenas a correspondência de palavras.

## Abordagens Implementadas

### Abordagem 1: Similaridade Direta com TF-IDF

Este modelo representa a abordagem clássica.
-   **Vetorização:** Cada texto é convertido em um vetor numérico usando a técnica **TF-IDF** (Term Frequency-Inverse Document Frequency).
-   **Comparação:** A **Similaridade de Cosseno** é calculada para medir o ângulo entre os vetores.

Este método é excelente para detectar cópias diretas ("copia e cola"), mas falha significativamente em casos de paráfrases, onde o significado é mantido com palavras diferentes.

### Abordagem 2: Classificador com Machine Learning

Esta abordagem trata a detecção de plágio como um problema de **classificação binária** ("é plágio" ou "não é plágio").
1.  **Engenharia de Features:** O script `src/2_generate_features.py` calcula a **similaridade TF-IDF** para milhares de pares de textos, usando-a como uma "feature" numérica.
2.  **Treinamento:** Um modelo de **Regressão Logística** é treinado (`src/2_train_classifier.py`) com essas features para aprender um limiar de decisão otimizado.
3.  **Predição:** O modelo treinado classifica novos pares de textos, oferecendo uma decisão mais robusta que um limiar fixo.

### Abordagem 3: Redes Siamesas com Transformers (SAMV3)

Esta é a abordagem mais avançada, focada em entender a **semelhança semântica**.
1.  **Arquitetura:** Utiliza uma **Rede Neural Siamesa** com um modelo **Transformer** pré-treinado (`sentence-transformers/all-mpnet-base-v2`) como "cérebro". O objetivo é aprender a mapear sentenças para um espaço vetorial onde a distância representa o significado.
2.  **Fine-Tuning:** O modelo pré-treinado passa por um treinamento adicional (*fine-tuning*) com o dataset **STS-B (Semantic Textual Similarity Benchmark)**. Usando `CosineSimilarityLoss`, o modelo aprende a produzir scores de similaridade graduados e com nuances (conforme script `src/train_sts.py`).
3.  **Análise:** Este modelo compara o texto suspeito com **todos** os documentos da base de dados, um por um, para encontrar a correspondência com a maior similaridade semântica, garantindo uma análise completa, embora computacionalmente intensiva.

## Como Executar a Aplicação Web (Streamlit)

Para executar a interface interativa e testar os modelos, siga os passos abaixo.

### Pré-requisitos
-   Python 3.8+
-   Git

### Instalação

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/CesarGuimaraes2000/Nocopi.git](https://github.com/CesarGuimaraes2000/Nocopi.git)
    cd Nocopi
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    # Cria o ambiente
    python -m venv .venv

    # Ativa o ambiente (Windows)
    .venv\Scripts\activate

    # Ativa o ambiente (macOS/Linux)
    source .venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

### Executando a Aplicação

Após a instalação, inicie a aplicação com o seguinte comando. O script se encarregará de baixar os pacotes de dados necessários da biblioteca NLTK na primeira execução.

```bash
streamlit run src/app.py
```

Isso abrirá uma nova aba no seu navegador com a interface da aplicação, onde você poderá enviar um arquivo `.txt`, selecionar um dos modelos e ver o resultado da análise em tempo real.

---

## (Opcional) Retreinando os Modelos

Se desejar executar o pipeline de treinamento do zero ou experimentar com diferentes parâmetros, siga os passos abaixo. Recomenda-se o uso de um ambiente com GPU (como o Google Colab) para treinar o Modelo 3.

1.  **Para o Modelo 2:**
    -   Execute `python src/2_generate_features.py` para criar o `features.csv`.
    -   Execute `python src/2_train_classifier.py` para treinar o modelo de Regressão Logística.

2.  **Para o Modelo 3 (SAMV3):**
    -   **Configure o Treinamento:** Abra o arquivo `config.json` (para treinar com STS-B) ou `config_parasci.json` (para treinar com ParaSCI). Neste arquivo, você pode ajustar os hiperparâmetros do treinamento:
        -   `"model_name"`: O modelo base do Hugging Face.
        -   `"batch_size"`: Quantos exemplos processar de uma vez (afeta o uso de memória).
        -   `"epochs"`: Quantas vezes o modelo verá o dataset inteiro.
        -   `"learning_rate"`: O "tamanho do passo" do aprendizado.
    -   **Execute o Script de Treino:** Para treinar, rode:
        ```bash
        python 3_train_siamese_model.py
        ```