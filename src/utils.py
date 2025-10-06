import streamlit as st
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def verificar_e_baixar_nltk():
    try:
        # Tenta usar os pacotes
        nltk.corpus.stopwords.words('english')
        nltk.word_tokenize('test sentence')
    except LookupError:
        # Se falhar, baixa
        st.info("Baixando pacotes de dados necessários do NLTK...")
        nltk.download('stopwords')
        nltk.download('punkt')
        st.success("Pacotes do NLTK baixados com sucesso!")
        st.rerun() 

def preprocessar_texto(texto):
    """Limpa e prepara o texto para análise."""
    stopwords = nltk.corpus.stopwords.words('english')
    texto = texto.lower()
    texto = re.sub(r'[^a-zA-Z\s]', repl='', string=texto, flags=re.I|re.A)
    tokens = nltk.word_tokenize(texto)
    tokens_filtrados = [palavra for palavra in tokens if palavra not in stopwords]
    return " ".join(tokens_filtrados)

def calcular_similaridade_tfidf(texto1, texto2):
    """Calcula a similaridade de cosseno TF-IDF entre dois textos."""
    vectorizer = TfidfVectorizer()
    textos_processados = [preprocessar_texto(texto1), preprocessar_texto(texto2)]
    if not textos_processados[0] or not textos_processados[1]:
        return 0.0
    matriz_tfidf = vectorizer.fit_transform(textos_processados)
    return cosine_similarity(matriz_tfidf[0:1], matriz_tfidf[1:2])[0][0]
