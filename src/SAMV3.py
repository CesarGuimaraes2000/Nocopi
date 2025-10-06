from sentence_transformers import util
# Precisamos importar a função de cálculo do TF-IDF novamente
from utils import calcular_similaridade_tfidf

def analisar_com_samv3(texto_suspeito, base_dados_originais, modelo_samv3, limiar_final=0.80, **kwargs):
    """
    Executa a análise semântica completa e também calcula a similaridade lexical
    para uma comparação mais rica.
    """
    if modelo_samv3 is None:
        return {"erro": "Modelo SAMV3 não foi carregado."}

    melhor_match = {
        "plagio_detectado": False,
        "fonte_provavel": None,
        "similaridade_semantica": 0.0,
        "similaridade_tfidf": 0.0 
    }

    # Prepara o embedding do texto suspeito (uma única vez)
    embedding_suspeito = modelo_samv3.encode(texto_suspeito, convert_to_tensor=True)

    # Itera sobre TODOS os documentos originais
    for nome_original, texto_original in base_dados_originais.items():
        # Gera o embedding para o texto original atual
        embedding_original = modelo_samv3.encode(texto_original, convert_to_tensor=True)
        
        # Calcula a similaridade semântica
        similaridade_semantica = util.cos_sim(embedding_suspeito, embedding_original).item()
        
        # Guarda o melhor resultado encontrado com base na similaridade SEMÂNTICA
        if similaridade_semantica > melhor_match["similaridade_semantica"]:
            # --- CORREÇÃO: CALCULA O TF-IDF APENAS PARA O MELHOR CANDIDATO ---
            # Isso é mais eficiente do que calcular para todos.
            similaridade_tfidf = calcular_similaridade_tfidf(texto_suspeito, texto_original)

            melhor_match["fonte_provavel"] = nome_original
            melhor_match["similaridade_semantica"] = similaridade_semantica
            melhor_match["similaridade_tfidf"] = similaridade_tfidf

    # Avalia o resultado final com base na maior similaridade encontrada
    if melhor_match["similaridade_semantica"] > limiar_final:
        melhor_match["plagio_detectado"] = True
            
    return melhor_match