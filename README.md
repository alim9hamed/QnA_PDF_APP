# **QnA PDF App using Retrieval-Augmented Generation (RAG)**

This project implements a **QnA PDF App** using **Retrieval-Augmented Generation (RAG)**, a powerful technique that enhances the capabilities of language models by allowing them to access relevant information from external documents. The app leverages a combination of **Groq LLM**, **Chroma vector database**, and **Gradio interface** to provide answers to user queries based on the content of an uploaded PDF document.

## **Table of Contents**

1. [Overview](#overview)
2. [Technologies Used](#technologies-used)
3. [Installation Instructions](#installation-instructions)
4. [How It Works](#how-it-works)
5. [API Usage](#api-usage)
6. [Gradio Interface](#gradio-interface)
7. [Environment Setup](#environment-setup)
8. [Contributing](#contributing)

## **Overview**

The **QnA PDF App** allows users to upload a PDF document and ask questions related to its content. The app performs **Retrieval-Augmented Generation (RAG)** to answer questions by searching for relevant text in the uploaded PDF, retrieving similar documents, and generating responses using a **Groq-powered model**. This app is designed to handle large PDFs and provides an interactive interface for users to engage with the document.

## **Technologies Used**

- **Groq**: A hardware-accelerated platform for AI, used for running the language model (LLama-3.3-70b).
- **Langchain**: A framework to handle natural language processing tasks and chains for question answering.
- **Chroma**: A vector database for storing and retrieving document embeddings.
- **NLTK**: A Python library used for text processing, including tokenization and text splitting.
- **HuggingFace Embeddings**: Pre-trained embeddings model for text vectorization.
- **Gradio**: A Python library for creating user-friendly web interfaces for machine learning applications.
- **Python-dotenv**: A library to load environment variables from `.env` files.

## **Installation Instructions**

### **Prerequisites**
Before you begin, ensure you have Python 3.7 or later installed.

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/alim9hamed/AI-Projects.git
cd AI-Projects
```

### **Step 2: Install Required Dependencies**

Install the required Python libraries using pip:

```bash
pip install -r requirements.txt
```

### **Step 3: Set Up the .env File**

Create a `.env` file in the root of the project and add your **Groq API key**:

```
GROQ_API_KEY=your_groq_api_key_here
```

### **Step 4: Download NLTK Resources**

The application uses NLTK for text splitting, which requires downloading certain resources. This is done automatically when the app is run.

### **Step 5: Launch the Application**

Run the Gradio interface:

```bash
python app.py
```

Once the server starts, navigate to the provided URL (usually `http://localhost:7860`) in your browser to interact with the app.

## **How It Works**

1. **PDF Upload**: Users can upload a PDF file containing the content they want to query.
2. **Text Processing**: The app loads the PDF and splits its text into manageable chunks using NLTK's text splitter.
3. **Vector Database Creation**: The text chunks are then embedded into vectors using **HuggingFace's embedding model**, and stored in **Chroma** for efficient retrieval.
4. **Question Answering**: When a user asks a question, the system performs a similarity search in the vector database, retrieving relevant text. The **Groq-powered model** processes the retrieved context and generates an answer.
5. **Interactive Q&A**: The result is displayed in an interactive Gradio interface where users can ask questions in real time.

## **API Usage**

### **RAGApp Class**

The `RAGApp` class handles the core functionality of the app.

#### **Methods**:

- `__init__(self, groq_api_key, pdf_file, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2")`: Initializes the app with the Groq API key, PDF file, and embedding model.
- `process_document(self)`: Processes the uploaded PDF by splitting the text into chunks and creating the vector database.
- `generate_answer(self, question)`: Uses the vector database and Groq model to generate an answer to the provided question.

### **get_answer Function**

This function interacts with the `RAGApp` class to generate answers based on the user's question and the uploaded PDF.

```python
def get_answer(question, pdf_file):
    rag_app = RAGApp(groq_api_key=GROQ_API_KEY, pdf_file=pdf_file.name)
    return rag_app.generate_answer(question)
```

## **Gradio Interface**

The app's interface is built using **Gradio**, allowing users to interact with the system via a web interface. The following components are provided:

- **PDF Upload**: Allows users to upload a PDF document.
- **Textbox for Questions**: Users can input questions related to the uploaded PDF.
- **Search Button**: A button to submit the question and process the answer.
- **Answer Display**: The system's response is shown in a text box.

```python
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

demo.launch()
```

## **Environment Setup**

### **Required Environment Variables**

Ensure that the following environment variables are set in your `.env` file:

- `GROQ_API_KEY`: Your Groq API key for accessing the Groq model.

### **Dependencies**

- Python 3.7+
- Libraries listed in `requirements.txt`

### **Running the Application**

Simply run the Python script `app.py` to start the Gradio interface 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QpSX0MZtvbSWwqCubRFlqao5A_2PcQ34?usp=sharing)

## **Contributing**

If you'd like to contribute to this project, feel free to fork the repository, create a branch, and submit a pull request. Please ensure that your code follows the existing structure and includes necessary tests.
