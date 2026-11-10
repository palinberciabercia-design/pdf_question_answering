
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# STEP 1: Load PDF chunks
df = pd.read_csv("dataset/chunks.csv")

documents = df["text"].fillna("").tolist()

# STEP 2: Prepare 15 questions
queries = [
    "What is the Transformer architecture?",
    "Why does the Transformer use self-attention?",
    "What are the limitations of recurrent neural networks?",
    "How does the encoder-decoder architecture work?",
    "What is scaled dot-product attention?",
    "What is multi-head attention?",
    "What is positional encoding?",
    "How does the Transformer encoder work?",
    "How does the Transformer decoder work?",
    "Why is masking used in the decoder?",
    "What datasets were used to train the Transformer?",
    "Which optimizer was used to train the model?",
    "What is label smoothing?",
    "How does the Transformer improve parallelization?",
    "What were the experimental results of the Transformer?"
]

# STEP 3: Convert documents into TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english")
document_vectors = vectorizer.fit_transform(documents)

# STEP 4: Perform keyword-based search
keyword_results = []

for query in queries:

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        document_vectors
    ).flatten()

    top_indices = scores.argsort()[::-1][:5]

    for rank, index in enumerate(top_indices, start=1):

        keyword_results.append({
            "query": query,
            "rank": rank,
            "document": documents[index],
            "similarity_score": round(float(scores[index]), 4)
        })

    print("Completed:", query)

# STEP 5: Save keyword search results
keyword_df = pd.DataFrame(keyword_results)

keyword_df.to_csv(
    "dataset/keyword_search_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nKeyword search completed!")
print("Total results:", len(keyword_results))
print("Saved to dataset/keyword_search_results.csv")

# STEP 6: Compare both search methods
semantic_df = pd.read_csv("dataset/search_results.csv")

print("\nSEMANTIC SEARCH RESULTS:", len(semantic_df))
print("KEYWORD SEARCH RESULTS:", len(keyword_df))

print("\nComparison completed successfully!")
