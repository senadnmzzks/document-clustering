# Document Clustering

A small app that groups .txt files by their meaning. Documents about the same topic get grouped together, even if they use different words.

## How it works

1. Each document is turned into vectors with an embedding model.
2. AgglomerativeClustering compares the vectors by cosine distance and groups documents together if their similarity is above the threshold.
3. The Qwen model then names the groups and writes a summary with how many groups there are and their details.

## Setup

​```bash
pip install -r requirements.txt
ollama pull qwen2.5
​```

## Usage

​```bash
streamlit run document_clustering.py
​```

Upload your `.txt` files, set the similarity threshold, and click **Cluster**.

## Tech stack

Python · sentence-transformers · Streamlit · Ollama + Qwen
