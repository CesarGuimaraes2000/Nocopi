# Dentro de src/SAMV1.py
from utils import calcular_similaridade_tfidf

def analisar_com_tfidf(texto_suspeito, base_dados_originais, limiar=0.85):
    """
    Compara um texto suspeito com uma base de dados, encontra a maior
    similaridade e avalia se ultrapassa o limiar.
    """
    melhor_match = {"fonte_provavel": None, "similaridade": 0.0}

    for nome_original, texto_original in base_dados_originais.items():
        similaridade = calcular_similaridade_tfidf(texto_suspeito, texto_original)
        
        # Guarda a maior similaridade encontrada até agora
        if similaridade > melhor_match["similaridade"]:
            melhor_match["fonte_provavel"] = nome_original
            melhor_match["similaridade"] = similaridade
            
    # Avalia o resultado final após checar todos os arquivos
    plagio_detectado = melhor_match["similaridade"] > limiar
    
    return {
        "plagio_detectado": plagio_detectado,
        "fonte_provavel": melhor_match["fonte_provavel"],
        "similaridade": melhor_match["similaridade"]
    }