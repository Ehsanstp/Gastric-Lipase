# GASTRIC-LIPASE
**Digest your big fat textbooks**

A fast, local, privacy-first RAG chatbot that lets you chat with your PDFs and textbooks using **Ollama** + **ChromaDB**.

Optimized to run efficiently even on machines with limited RAM.

## Features

- Fully local inference with Ollama
- Efficient PDF processing and chunking
- Vector database using ChromaDB
- Chat history stored in Neon PostgreSQL
- Memory-efficient ingestion pipeline
- Highly configurable

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/Ehsanstp/Gastric-lipase.git
    cd gastric-lipase
    ```

2. Instal dependencies
    ```bash
    pip install -r requirements.txt
    ```

4. Set up Neon Database
    - Create a free PostgreSQL database at Neon
    - Run the SQL commands from `schema.sql` in your [Neon](https://neon.com/) SQL editor 
    - Copy the generated `DATABASE_URL`

4. Configure environment variables
    ```bash
    nano .env
    ```
    Then paste your Neon `DATABASE_URL` into the `.env` file.

5. Create a folder `data` to add your pdfs
     ```bash
     mkdir data
     ```
     Add all the books as pdfs inside `./data/` folder. And modify changeable parameters in `config.py`.

7. Run ollama using
     ```bash
     ollama serve
     ```

9. Run ingest.py once
    ```bash
    python ingest.py
    ``` 
    This will generate the chromadb vectorstore automatically and store the embeddings.

10. Run
    ```bash
    streamlit run app.py
    ```

11. Open http://localhost:8501 in a browser and start studying


**Get Ready to digest heavy textbooks like never before.**