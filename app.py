from fastapi import FastAPI
import chromadb
import ollama

app = FastAPI()
chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection("docs")

@app.post("/query")
def query(q: str):
    results = collection.query(query_texts=[q], n_results=1)
    context = results["documents"][0][0] if results["documents"] else ""

    answer = ollama.generate(
        model="tinyllama",
        prompt=f"Context:\n{context}\n\nQuestion: {q}\nAnswer clearly and concisely:",
    )

    return {"answer": answer["response"]}

@app.post("/add")
def add_knowledge(text: str):
    """Add knowledge to the ChromaDB collection"""
    try:
        # Generate a unique ID for the document
        import uuid
        doc_id = str(uuid.uuid4())

        # Add the text to the collection
        collection.add(documents=[text], ids=[doc_id])

        return {
            "status": "success",
            "message": "Knowledge added successfully",
            "id": doc_id
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }