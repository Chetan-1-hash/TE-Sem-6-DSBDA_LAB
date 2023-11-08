from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd

# Sample sentences
sentences = ["GeeksforGeeks is a Computer Science portal for Geeks.",
             "Science is the systematic study of the universe.",
             "I love watching science fiction movies."]

# Tokenization
tokens = [word_tokenize(sentence) for sentence in sentences]

# POS tagging
pos_tags = [pos_tag(token) for token in tokens]

# Stop word removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [[token for token in sentence_tokens if token.lower() not in stop_words]
                   for sentence_tokens in tokens]

# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [[stemmer.stem(token) for token in sentence_tokens]
                  for sentence_tokens in filtered_tokens]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [[lemmatizer.lemmatize(token) for token in sentence_tokens]
                     for sentence_tokens in filtered_tokens]

# Create a string with the preprocessed tokens
tokens_processed = [" ".join(lemmatized_sentence_tokens) for lemmatized_sentence_tokens in lemmatized_tokens]

# Calculate Term Frequency-Inverse Document Frequency (TF-IDF)
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(tokens_processed)
df_tfidf = pd.DataFrame(tfidf[0].T.todense(), index=vectorizer.get_feature_names_out(), columns=["tfidf"])
df_tfidf = df_tfidf.sort_values('tfidf', ascending=False)

# Calculate Term Frequency (TF)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

print("Original sentences:\n", sentences)
print("Tokens:\n", tokens)
print("POS tags:\n", pos_tags)
print("Filtered tokens (stop words removed):\n", filtered_tokens)
print("Stemmed tokens:\n", stemmed_tokens)
print("Lemmatized tokens:\n", lemmatized_tokens)
print("\nTerm Frequency-Inverse Document Frequency:\n", df_tfidf)
print("\nTerm Frequency:\n", X.toarray())
print("Feature names:\n", vectorizer.get_feature_names_out())
