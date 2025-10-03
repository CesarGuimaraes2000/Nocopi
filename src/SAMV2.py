import pandas as pd
from utils import calcular_similaridade_tfidf

def analisar_com_ml(texto_suspeito, base_dados_originais, modelo_ml):
    """
    Compara um texto suspeito com toda a base de dados, encontra a maior
    similaridade e a maior evidência de plágio usando o modelo de ML.
    """
    if modelo_ml is None:
        return {"erro": "Modelo de Machine Learning não foi carregado."}

    # Dicionário para guardar a melhor correspondência geral (maior similaridade)
    melhor_match_geral = {"fonte_provavel": None, "similaridade": -1.0}
    
    # Dicionário para guardar a melhor detecção de plágio (maior confiança)
    melhor_match_plagio = {"fonte_provavel": None, "similaridade": -1.0, "confianca": -1.0}
    
    plagio_detectado_em_algum_momento = False

    # Itera sobre a base de dados UMA ÚNICA VEZ
    for nome_original, texto_original in base_dados_originais.items():
        similaridade = calcular_similaridade_tfidf(texto_suspeito, texto_original)
        
        # 1. Guarda sempre a maior similaridade encontrada
        if similaridade > melhor_match_geral["similaridade"]:
            melhor_match_geral["fonte_provavel"] = nome_original
            melhor_match_geral["similaridade"] = similaridade

        # 2. Usa o modelo de ML para fazer a predição em TODOS os pares
        dados_para_previsao = pd.DataFrame({'similaridade_tfidf': [similaridade]})
        predicao = modelo_ml.predict(dados_para_previsao)[0]
        
        # 3. Se o modelo detectar plágio, verifica se é a melhor detecção até agora
        if predicao == 1:
            plagio_detectado_em_algum_momento = True
            probabilidades = modelo_ml.predict_proba(dados_para_previsao)
            confianca = probabilidades[0][1] # Confiança na classe "plágio"
            
            if confianca > melhor_match_plagio["confianca"]:
                melhor_match_plagio["fonte_provavel"] = nome_original
                melhor_match_plagio["similaridade"] = similaridade
                melhor_match_plagio["confianca"] = confianca

    # Após o loop, decide o que retornar
    if plagio_detectado_em_algum_momento:
        # Se encontrou plágio, retorna os detalhes da detecção de maior confiança
        return {
            "plagio_detectado": True,
            "fonte_provavel": melhor_match_plagio["fonte_provavel"],
            "similaridade": melhor_match_plagio["similaridade"],
            "confianca": melhor_match_plagio["confianca"]
        }
    else:
        # Se não encontrou plágio, retorna os detalhes da maior similaridade geral
        return {
            "plagio_detectado": False,
            "fonte_provavel": melhor_match_geral["fonte_provavel"],
            "similaridade": melhor_match_geral["similaridade"]
        }