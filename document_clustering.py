import streamlit as st
from sentence_transformers import SentenceTransformer, util
import requests

EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen2.5"

# Cache the model so it isn't reloaded every time
@st.cache_resource
def load_model():
    return SentenceTransformer(EMBED_MODEL)

model = load_model()

# Sends the prompt to the model and returns the reply as a single chunk of text (stream=False)
def ask_llm(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
    ).json()
    return response["response"].strip()

# Streamlit UI ; page title and the description text shown on screen
st.title("📄 Document Clustering")
st.write("Upload documents; the app groups them by topic and names each group.")

# Streamlit file uploader: accepts .txt only, allows multiple files
uploaded_files = st.file_uploader(
    "Select document files (.txt)",
    type="txt",
    accept_multiple_files=True,
)

# Sets the min, max and default values of the similarity threshold
threshold = st.slider("Similarity threshold (higher = stricter grouping)", 0.30, 0.90, 0.55, 0.05)

if st.button("Cluster"):

    if not uploaded_files:
        st.warning("Please upload at least two documents.")
        st.stop()

    # Reads the uploaded files' names and contents into two separate lists
    filenames = [f.name for f in uploaded_files]
    documents = [f.read().decode("utf-8").strip() for f in uploaded_files]

    # The model runs here and creates the vectors of the documents
    with st.spinner("Analyzing documents..."):
        embeddings = model.encode(documents, convert_to_tensor=True)

        parent = list(range(len(filenames)))
        # find: returns which group a document belongs to  
        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x
        # union: merges two documents into the same group
        def union(a, b):
            parent[find(a)] = find(b)
        # Groups together documents whose similarity score is above the threshold
        for i in range(len(filenames)):
            for j in range(i + 1, len(filenames)):
                similarity = float(util.cos_sim(embeddings[i], embeddings[j]))
                if similarity >= threshold:
                    union(i, j)
        # Collects documents that belong to the same group into one list
        cluster_map = {}
        for i in range(len(filenames)):
            cluster_map.setdefault(find(i), []).append(i)
        clusters = list(cluster_map.values())
    # Shows the number of clusters found on screen
    st.subheader(f"{len(clusters)} clusters found")

    cluster_info = []
    for n, members in enumerate(clusters, start=1):
        text = "\n\n".join(documents[i] for i in members)
        # Sends the collected document contents to the model and asks for a topic name based on our prompt
        prompt = (f"The following texts belong to the same topic. State the common topic "
                  f"in at most 3 words, giving only the topic name.\n\nTexts:\n{text}\n\nTopic:")
        with st.spinner(f"Labeling cluster {n}..."):
            label = ask_llm(prompt)

        cluster_info.append((len(members), label))
        with st.expander(f"Cluster {n}: {label}  ({len(members)} docs)"):
            for i in members:
                st.markdown(f"**{filenames[i]}**")
                st.write(documents[i])
    # Builds a list of the group info, prepares the summary prompt, and sends it to the model
    lines = "\n".join(f"- {count} docs: {label}" for count, label in cluster_info)
    summary_prompt = (
        "Below are document groups with their topics and counts. "
        "Combine them into one fluent English summary sentence. "
        'Example: "2 documents about social media, 1 document about leave processes."\n\n'
        f"Data:\n{lines}\n\n"
        "Summary:"
    )
    with st.spinner("Writing summary..."):
        summary = ask_llm(summary_prompt)

    st.success(summary)