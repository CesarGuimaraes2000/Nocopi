# Detector de Plágio com Inteligência Artificial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CesarGuimaraes2000/Nocopi)

Este projeto foi desenvolvido como parte da disciplina de Inteligência Artificial. O objetivo é implementar duas abordagens distintas para a detecção de plágio em textos, utilizando um dataset público para treinar e validar os modelos.

O projeto culmina em uma aplicação web interativa construída com Streamlit, que permite ao usuário testar ambos os modelos.

## Metodologia

O projeto implementa duas metodologias principais para a detecção de plágio, representando diferentes eras e abordagens da Inteligência Artificial:

1.  **IA Clássica (Recuperação de Informação):** Uma abordagem determinística baseada em estatística e álgebra linear.
2.  **Aprendizado de Máquina Supervisionado (Classificação):** Uma abordagem onde um modelo "aprende" a partir de exemplos rotulados para tomar decisões.

## Abordagens Implementadas

### Abordagem 1: Similaridade Direta com TF-IDF

Este modelo representa a abordagem clássica. O processo consiste em:

  - **Vetorização:** Cada texto é convertido em um vetor numérico usando a técnica **TF-IDF** (Term Frequency-Inverse Document Frequency), que mede a importância de cada palavra no texto em relação a uma coleção de documentos.
  - **Comparação:** A **Similaridade de Cosseno** é calculada para medir o ângulo entre os vetores de dois textos. Uma similaridade próxima de 100% indica que os textos são muito parecidos.

Este método é excelente para detectar cópias diretas ("copia e cola"), mas é menos eficaz com paráfrases complexas, pois não captura o significado semântico do texto.

### Abordagem 2: Classificador com Machine Learning

Esta abordagem trata a detecção de plágio como um problema de **classificação binária** ("é plágio" ou "não é plágio"). O fluxo de trabalho é:

1.  **Engenharia de Features:** Um script (`src/1_generate_features.py`) processa pares de textos (um original e um suspeito) e extrai uma característica numérica (feature) para cada par: a própria **similaridade TF-IDF**. O resultado é salvo em `features.csv`.
2.  **Treinamento:** Um modelo de **Regressão Logística** é treinado (`src/2_train_classifier.py`) com os dados do `features.csv`. O modelo aprende a "regra" que separa os casos de plágio (`rótulo=1`) dos não-plágios (`rótulo=0`) com base no valor da similaridade.
3.  **Predição:** O modelo treinado e salvo (`plagiarism_classifier.joblib`) é usado na aplicação final para classificar novos pares de textos, oferecendo uma decisão "sim" ou "não" com um grau de confiança.

## Como Executar a Aplicação Web (Streamlit)

Para executar a interface interativa e testar os modelos, siga os passos abaixo.

### Pré-requisitos

  - Python 3.8 ou superior
  - Git

### Instalação

1.  **Clone o repositório:**

    ```bash
    git clone https://github.com/CesarGuimaraes2000/Nocopi
    cd [NOME_DA_PASTA_DO_PROJETO]
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

Isso abrirá uma nova aba no seu navegador com a interface da aplicação, onde você poderá enviar um arquivo `.txt`, selecionar um dos dois modelos e ver o resultado da análise em tempo real.
