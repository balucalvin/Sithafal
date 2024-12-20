import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class PDFRAGPipeline:
    def __init__(self, pdf_path ):
       
        self.text = self.extract_text_from_pdf(pdf_path)
        
        
        self.chunks = self.chunk_text(self.text)
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(self.chunks, convert_to_tensor=True)
        
        
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])  
        self.index.add(np.array([e.cpu().detach().numpy() for e in self.embeddings]))
        
        
        self.generator = pipeline('text-generation', model='gpt-3.5-turbo')

    def extract_text_from_pdf(self, pdf_path):
       
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text

    def chunk_text (self, text, chunk_size=500):
        words = text.split()
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks

    def retrieve_relevant_chunks(self, query, top_k=5):
      
        query_embedding = self.model.encode(query, convert_to_tensor=True).cpu().detach().numpy()
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        return [self.chunks[idx] for idx in indices[0]]

    def generate_answer(self, query):
       
        relevant_chunks = self.retrieve_relevant_chunks(query)
        context = ' '.join(relevant_chunks)
        prompt = f"Context: {context}\n\nAnswer the following question: {query}"
        return self.generator(prompt, max_length=200)[0]['generated_text']


pdf_path = 'C:\MERN\1.RESUME.pdf'  
pipeline = PDFRAGPipeline(pdf_path)

query = "Explain the key points in the document."
response = pipeline.generate_answer(query)
print(response)
