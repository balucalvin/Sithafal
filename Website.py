import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

class WebsiteRAGPipeline:
    def __init__(self, url):
        # Step 1: Fetch and Ingest Website Content
        self.url = url
        self.text = self.fetch_website_content()
        
        # Step 2: Preprocess and Chunk Text
        self.chunks = self.chunk_text(self.text)
        
        # Step 3: Create Embeddings for the Chunks
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(self.chunks, convert_to_tensor=True)
        
        # Step 4: Store the Embeddings in FAISS for Retrieval
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])  # L2 distance
        self.index.add(np.array([e.cpu().detach().numpy() for e in self.embeddings]))
        
        # Step 5: Load the Generative Model (e.g., GPT-3.5-turbo)
        self.generator = pipeline('text-generation', model='gpt-3.5-turbo')

    def fetch_website_content(self):
        """Fetches the HTML content of the website and returns the text."""
        response = requests.get(self.url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract meaningful content (e.g., from paragraphs, headings, lists, etc.)
        content = ''
        for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'li']):  # Add more tags if needed
            content += tag.get_text(separator=' ') + ' '
        
        return content

    def chunk_text(self, text, chunk_size=500):
        """Splits the website content into smaller chunks for retrieval."""
        words = text.split()
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks

    def retrieve_relevant_chunks(self, query, top_k=5):
        """Retrieves the top_k most relevant chunks for the given query."""
        query_embedding = self.model.encode(query, convert_to_tensor=True).cpu().detach().numpy()
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        return [self.chunks[idx] for idx in indices[0]]

    def generate_answer(self, query):
        """Generates an answer based on the query using the retrieved relevant chunks."""
        relevant_chunks = self.retrieve_relevant_chunks(query)
        
        # Combine relevant chunks into a single context
        context = ' '.join(relevant_chunks)
        
        # Generate a response using the generative model
        prompt = f"Context: {context}\n\nAnswer the following question: {query}"
        generated_text = self.generator(prompt, max_length=200)[0]['generated_text']
        
        return generated_text

# Example Usage
if __name__ == '__main__':
    # Step 1: Initialize the pipeline with the website URL
    url = 'https://facebook.com'  # Replace with the actual website URL
    pipeline = WebsiteRAGPipeline(url)

    # Step 2: Input your query
    query = "What services does the company provide?"

    # Step 3: Generate the response based on the query
    response = pipeline.generate_answer(query)
    
    # Step 4: Output the generated response
    print("Query:", query)
    print("Response:", response)
