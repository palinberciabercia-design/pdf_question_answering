
from pypdf import PdfReader
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

# STEP 1: Extract text from PDF
pdf_path = "dataset/sample.pdf"
reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"

print("Total Pages:", len(reader.pages))
print("Total Characters:", len(text))

# Save extracted text
with open("dataset/extracted_text.txt", "w", encoding="utf-8") as file:
    file.write(text)

print("PDF text extracted successfully!")

# STEP 2: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_text(text)

# STEP 3: Save chunks into CSV
df = pd.DataFrame({
    "chunk_id": range(1, len(chunks) + 1),
    "text": chunks
})

df.to_csv("dataset/chunks.csv", index=False)

print("Total chunks:", len(chunks))
print("Chunks saved successfully!")

# STEP 4: Generate Embeddings and Store in ChromaDB

import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for chunks
embeddings = model.encode(chunks).tolist()

print("Embeddings generated:", len(embeddings))

# Create persistent ChromaDB database
client = chromadb.PersistentClient(path="chroma_db")

# Recreate collection to avoid duplicate data
try:
    client.delete_collection("pdf_collection")
except Exception:
    pass

collection = client.create_collection(
    name="pdf_collection",
    metadata={"hnsw:space": "cosine"}
)

# Store chunks and embeddings
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings,
    metadatas=[{"chunk_id": i} for i in range(len(chunks))]
)

print("Chunks stored in ChromaDB successfully!")
print("Total documents:", collection.count())

# STEP 5: Semantic Search

query_text = "What is the Transformer architecture?"

# Convert question into embedding
query_embedding = model.encode(query_text).tolist()

# Retrieve Top-5 relevant chunks
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

print("\nUser Query:", query_text)
print("\nTop-5 Relevant Documents:\n")

search_results = []

for i in range(5):
    document = results["documents"][0][i]
    distance = results["distances"][0][i]

    similarity_score = 1 - distance

    print("Rank:", i + 1)
    print("Similarity Score:", round(similarity_score, 4))
    print("Document:", document)
    print("-" * 50)

    search_results.append({
        "query": query_text,
        "rank": i + 1,
        "document": document,
        "similarity_score": similarity_score
    })

# Save results to CSV
search_df = pd.DataFrame(search_results)

search_df.to_csv(
    "dataset/search_results.csv",
    index=False
)

print("Search results saved successfully!")

# STEP 6: Retrieve Top-5 chunks for 15 questions

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

all_results = []

for query_text in queries:

    query_embedding = model.encode(query_text).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    for rank in range(len(results["documents"][0])):

        document = results["documents"][0][rank]
        distance = results["distances"][0][rank]

        similarity_score = 1 - distance

        all_results.append({
            "query": query_text,
            "rank": rank + 1,
            "document": document,
            "similarity_score": round(similarity_score, 4)
        })

    print("Completed:", query_text)

# Save all retrieval results
results_df = pd.DataFrame(all_results)

results_df.to_csv(
    "dataset/search_results.csv",
    index=False
)

print("\nAll queries completed!")
print("Total retrieved documents:", len(all_results))
print("Results saved to dataset/search_results.csv")

# STEP 7: PDF Question Answering using Ollama

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Load local LLM
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)

# User question
question = "What is the Transformer architecture?"

# Retrieve relevant chunks
query_embedding = model.encode(question).tolist()

retrieved = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

# Combine retrieved documents
context = "\n\n".join(retrieved["documents"][0])

# Create prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the provided context.
If the answer is not available in the context, say:
"I could not find the answer in the PDF."

Context:
{context}

Question:
{question}

Answer:
""")

# Generate answer
chain = prompt | llm

response = chain.invoke({
    "context": context,
    "question": question
})

print("\nQuestion:", question)
print("\nRetrieved Context:\n", context)
print("\nGenerated Answer:\n", response.content)

# STEP 8: Compare Answers With RAG and Without RAG

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Initialize Ollama
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)

# Create answer storage
answer_results = []

# Process all 15 questions
for question in queries:

    # WITHOUT RAG
    prompt_without_rag = f"""
    Answer the following question using your
    general knowledge.

    Question: {question}

    Answer:
    """

    response_without_rag = llm.invoke(prompt_without_rag)

    answer1 = response_without_rag.content

    # WITH RAG
    query_embedding = model.encode(question).tolist()

    retrieved = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    context = "\n\n".join(retrieved["documents"][0])

    prompt_with_rag = ChatPromptTemplate.from_template("""
    Answer the question using only the given context.
    If the answer is not available in the context,
    say "I could not find the answer in the PDF."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    chain = prompt_with_rag | llm

    response_with_rag = chain.invoke({
        "context": context,
        "question": question
    })

    answer2 = response_with_rag.content

    # Store both answers
    answer_results.append({
        "question": question,
        "answer_without_rag": answer1,
        "answer_with_rag": answer2,
        "retrieved_context": context
    })

    print("\nQuestion:", question)
    print("Without RAG:", answer1)
    print("With RAG:", answer2)
    print("-" * 60)

# Save answers to CSV
answers_df = pd.DataFrame(answer_results)

answers_df.to_csv(
    "dataset/answers.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nComparison completed!")
print("Answers saved to dataset/answers.csv")
print("Total questions:", len(answer_results))
# Save questions into CSV
queries_df = pd.DataFrame({
    "question_id": range(1, len(queries) + 1),
    "question": queries
})

queries_df.to_csv(
    "dataset/queries.csv",
    index=False
)

print("15 questions saved successfully!")