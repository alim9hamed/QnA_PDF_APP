# pip install  langchain gradio python-dotenv nltk chromadb langchain-groq huggingface_hub langchain-community sentence-transformers
import gradio as gr
import nltk
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import NLTKTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the Groq API key from the .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')

# RAG App Class
class RAGApp:
    def __init__(self, groq_api_key, pdf_file, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # Initialize Groq and other models
        self.llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Initialize PDF Loader and Text Splitter
        self.pdf_loader = PyPDFLoader(pdf_file)
        self.text_splitter = NLTKTextSplitter(chunk_size=300, chunk_overlap=50)
        
        # Initialize vector database
        self.vector_db_dir = "/content/pdf_chroma_db"
        self.vector_db = None

        # Load and process the document
        self.process_document()

    def process_document(self):
        # Load the document and split into chunks
        pages = self.pdf_loader.load_and_split()
        documents = [page.page_content for page in pages]
        metadatas = [{"document": f"Page {i+1}"} for i in range(len(pages))]

        # Split text into chunks
        tokens_chunks = self.text_splitter.create_documents(documents, metadatas=metadatas)

        # Create and store the vector database
        self.vector_db = Chroma.from_documents(tokens_chunks, self.embedding_model, persist_directory=self.vector_db_dir)

    def generate_answer(self, question):
        # Perform similarity search in the vector database
        similar_docs = self.vector_db.similarity_search(question, k=4)

        # Define the Q&A prompt template
        qna_template = "\n".join([
            "Answer the next question using the provided context.",
            "If the answer is not contained in the context, say 'NO ANSWER IS AVAILABLE'",
            "### Context:",
            "{context}",
            "",
            "### Question:",
            "{question}",
            "",
            "### Answer:",
        ])

        # Create the prompt template
        qna_prompt = PromptTemplate(template=qna_template, input_variables=['context', 'question'])

        # Load the QA chain with Groq
        stuff_chain = load_qa_chain(self.llm, chain_type="stuff", prompt=qna_prompt)

        # Get the answer using the QA chain
        answer = stuff_chain({
            "input_documents": similar_docs,
            "question": question
        }, return_only_outputs=True)

        return answer["output_text"]

# Define a function to interact with the Gradio interface
def get_answer(question, pdf_file):
    # Initialize the RAG app with the uploaded PDF and Groq API key from .env
    rag_app = RAGApp(
        groq_api_key=GROQ_API_KEY,  # Groq API key from environment
        pdf_file=pdf_file  # Path to the uploaded PDF file
    )
    
    # Get the answer to the question
    return rag_app.generate_answer(question)

import gradio as gr

# Create a more refined Gradio interface, using gr.themes.Soft instead of gr.themes.HuggingFace()
with gr.Blocks(theme=gr.themes.Soft()) as demo:  # Apply the Soft theme 
    gr.Markdown("# **QnA PDF App**")
    gr.Markdown(
        "This app uses **Retrieval-Augmented Generation (RAG)** to answer questions based on the provided PDF document."
    )
    
    with gr.Row():
        # PDF upload component
        pdf_upload = gr.File(label="Upload PDF", type="filepath")  # File upload component for PDF, changed type to "filepath"
        # Textbox for the user's question
        question_input = gr.Textbox(
            label="Enter your question", 
            placeholder="Ask your question here...",
            lines=2
        )
    
    with gr.Row():
        # Button to process the PDF and question
        submit_button = gr.Button("Search", variant="primary")
        
    # Output display
    answer_output = gr.Textbox(
        label="Answer", 
        placeholder="The answer will appear here.",
        lines=4,
        interactive=False
    )
    
    # Display additional helpful info or instructions
    gr.Markdown(
        "### Instructions:\n1. Upload a PDF file.\n2. Ask any question related to the content of the PDF."
    )
    
    # Define the interface functionality
    submit_button.click(fn=get_answer, inputs=[question_input, pdf_upload], outputs=answer_output)

# Launch the app
demo.launch()  # Remove the theme argument here