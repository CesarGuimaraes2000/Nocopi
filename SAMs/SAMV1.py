#importando livrarias para serem usadas
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

# Get a list of student files
# Make sure to put your .txt files in the same directory as this script,
# which is: C:/Users/Fagundes/OneDrive - ifsp.edu.br/Facul/8 - Semestre/IA/Copia não comedia/
student_file = [file for file in os.listdir() if file.endswith('.txt')]