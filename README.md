# Sithafal
Project tasks for sithafal
__init__ Method: Initializes the pipeline by:

Extracting text from the PDF.
Chunking the text.
Creating embeddings for the chunks and adding them to a FAISS index.
Initializing the generative language model (GPT-3.5-turbo in this case).
extract_text_from_pdf Method: Handles PDF text extraction using PyPDF2.

chunk_text Method: Splits the extracted text into smaller chunks for more efficient retrieval.

retrieve_relevant_chunks Method: Uses FAISS to retrieve the most relevant chunks based on a query by comparing embeddings.

generate_answer Method: Combines retrieved chunks as context and uses the GPT model to generate an answer based on the user's query.
