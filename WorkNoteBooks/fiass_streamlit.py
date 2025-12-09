import streamlit as st
import faiss
import numpy as np
from sklearn.manifold import TSNE
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ----------------------------
# Load FAISS index correctly
# ----------------------------

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

faiss_store = FAISS.load_local(
    "models/fiass/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

index = faiss_store.index
documents = [d.page_content for d in faiss_store.docstore._dict.values()]
metadatas = [d.metadata for d in faiss_store.docstore._dict.values()]

# ----------------------------

st.title("🔍 FAISS Embedding Viewer")

st.write("Total vectors:", index.ntotal)

# ----------------------------
# View individual vector
# ----------------------------

st.header("View Individual Vector")

vec_id = st.number_input("Vector ID", 0, index.ntotal - 1, 0)
vector = index.reconstruct(vec_id)

st.subheader("Embedding:")
st.write(vector)

st.subheader("Chunk Text:")
st.write(documents[vec_id])

st.subheader("Metadata:")
st.write(metadatas[vec_id])

# ----------------------------
# 2D Visualization using t-SNE
# ----------------------------

st.header("2D Visualization (t-SNE)")

if st.button("Generate 2D Plot"):
    all_vecs = np.array([index.reconstruct(i) for i in range(index.ntotal)])

    tsne = TSNE(n_components=2, learning_rate="auto", init="random").fit_transform(all_vecs)

    st.write("Plotting embeddings in 2D...")
    st.scatter_chart({"x": tsne[:, 0], "y": tsne[:, 1]})
