import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Mental Health Text Clustering",
    page_icon="🧠",
    layout="centered"
)

# Define stopword bigrams
STOPWORD_BIGRAMS = {
    "yang tidak", "tidak ada", "aku tidak", "tidak enak", "ada yang", 
    "tidak yang", "orang orang", "teman teman", "gara gara", "https chat", 
    "assalamu alaikum", "afc", "whatsapp com"
}

# Cluster mapping (model output -> your desired output)
CLUSTER_MAPPING = {
    0: 2,  # model cluster 0 -> your cluster 2s
    1: 3,  # model cluster 1 -> your cluster 3
    2: 1,  # model cluster 2 -> your cluster 1
    3: 0   # model cluster 3 -> your cluster 0
}

# Cluster topics
CLUSTER_TOPICS = {
    0: "Psychosomatics and early treatment of stress",
    1: "Meaning-seeking and alternative coping strategies", 
    2: "Existential concerns intertwined with religious and social pressures",
    3: "Emotional expression and relationship difficulties"
}

def remove_stopword_bigrams(text):
    """Remove stopword bigrams from text"""
    text = text.lower()
    for bigram in STOPWORD_BIGRAMS:
        text = text.replace(bigram, "")
    return text.strip()

def preprocess_text(text):
    """Preprocess input text"""
    cleaned_text = remove_stopword_bigrams(text)
    tokens = cleaned_text.split()
    return tokens

def document_vector(doc, w2v_model):
    """Convert document to vector using Word2Vec model"""
    vectors = [w2v_model.wv[word] for word in doc if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)

@st.cache_data
def load_cluster_data():
    """Load cluster CSV files"""
    try:
        all_data = []
        for i in range(4):
            file = f"cluster_{i}_resultsV7.csv"
            if os.path.exists(file):
                df = pd.read_csv(file)
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            return combined_df, True
        return None, False
    except Exception as e:
        st.error(f"Error loading cluster data: {e}")
        return None, False

@st.cache_resource
def load_models():
    """Load Word2Vec model and create KMeans"""
    try:
        # Load Word2Vec model
        w2v_model = Word2Vec.load("word2vec_mental_health.model")
        
        # Load cluster data and create KMeans
        cluster_df, _ = load_cluster_data()
        if cluster_df is None:
            return None, None, False
        
        # Prepare training data
        texts = cluster_df['tokenized'].fillna('').astype(str).tolist()
        texts = [remove_stopword_bigrams(text.lower()) for text in texts]
        tokenized_texts = [text.split() for text in texts if text.strip()]
        
        # Vectorize documents
        X_vectors = np.array([document_vector(doc, w2v_model) for doc in tokenized_texts])
        non_zero_mask = ~np.all(X_vectors == 0, axis=1)
        X_vectors_clean = X_vectors[non_zero_mask]
        
        # Create KMeans model
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(X_vectors_clean)
        
        return w2v_model, kmeans, True
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, False

def main():
    st.title("🧠 Mental Health Text Classification")
    st.markdown("### Classify Indonesian mental health text into thematic categories")
    
    # Load models
    w2v_model, kmeans_model, models_loaded = load_models()
    
    if not models_loaded:
        st.error("❌ Could not load models. Please ensure all required files are present.")
        st.stop()
    
    st.success("✅ Models loaded successfully!")
    
    # Text input
    st.subheader("📝 Enter Text for Classification")
    user_text = st.text_area(
        "Type Indonesian text here:",
        height=120,
        placeholder="Masukkan teks bahasa Indonesia di sini..."
    )
    
    # Classify button
    if st.button("🔍 Classify", type="primary"):
        if user_text.strip():
            # Preprocess text
            tokens = preprocess_text(user_text)
            
            if len(tokens) == 0:
                st.warning("⚠️ No valid words found after preprocessing.")
                return
            
            # Convert to vector
            doc_vector = document_vector(tokens, w2v_model)
            
            if np.all(doc_vector == 0):
                st.warning("⚠️ No words from your text found in vocabulary.")
                return
            
            # Predict cluster (model output)
            model_cluster = kmeans_model.predict([doc_vector])[0]
            
            # Map to your desired cluster
            final_cluster = CLUSTER_MAPPING[model_cluster]
            
            # Get topic
            topic = CLUSTER_TOPICS[final_cluster]
            
            # Display result
            st.subheader("📊 Classification Result")
            
            # Show the mapped cluster and topic
            st.markdown(f"**Cluster:** {final_cluster}")
            st.markdown(f"**Topic:** {topic}")
            
        else:
            st.warning("⚠️ Please enter some text to classify.")
    
    # Show cluster information
    st.markdown("---")
    st.subheader("📋 Cluster Topics")
    
    for cluster_id, topic in CLUSTER_TOPICS.items():
        st.markdown(f"**Cluster {cluster_id}:** {topic}")

if __name__ == "__main__":
    main()