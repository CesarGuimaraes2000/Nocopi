# src/test_sbert.py

from sentence_transformers import SentenceTransformer, util

# 1. Carrega o modelo pré-treinado escolhido do Hugging Face Hub
# Na primeira vez, ele será baixado e salvo em cache, o que pode demorar um pouco.
print("Carregando o modelo 'all-mpnet-base-v2'...")
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# 2. Define um par de sentenças semanticamente similares, mas lexicalmente diferentes
sentenca1 = "O conceito de herança permite formar novas classes usando classes que já foram definidas."
sentenca2 = "Herança é um mecanismo para reutilizar código de classes existentes com pouca ou nenhuma modificação."

# 3. Define uma sentença não relacionada
sentenca3 = "A programação dinâmica é uma técnica para resolver problemas complexos."

# Imprime as sentenças para referência
print("\nSentença 1:", sentenca1)
print("Sentença 2:", sentenca2)
print("Sentença 3:", sentenca3)

# 4. Gera os embeddings (vetores numéricos) para cada sentença
# Cada embedding é um vetor de alta dimensão que representa o significado da sentença.
embedding1 = model.encode(sentenca1, convert_to_tensor=True)
embedding2 = model.encode(sentenca2, convert_to_tensor=True)
embedding3 = model.encode(sentenca3, convert_to_tensor=True)

# 5. Calcula a similaridade de cosseno entre os embeddings
# A biblioteca `util` do sentence-transformers facilita este cálculo.
similaridade_1_2 = util.cos_sim(embedding1, embedding2)
similaridade_1_3 = util.cos_sim(embedding1, embedding3)

# 6. Exibe os resultados
print("\n--- Resultados da Análise Semântica ---")
print(f"Similaridade entre Sentença 1 e 2 (relacionadas): {similaridade_1_2.item():.4f}")
print(f"Similaridade entre Sentença 1 e 3 (não relacionadas): {similaridade_1_3.item():.4f}")