# LLM Question-Answering Application 🤖

This repository contains a Question-Answering (QA) system based on Large Language Models (LLM) using Streamlit for the frontend and LangChain with Ollama embeddings for document processing. This application allows users to upload documents in `.txt`, `.pdf`, and `.docx` formats, and then ask questions based on the content of the uploaded documents. The LLM will retrieve relevant information and provide answers.

## Features
- Upload documents (PDF, TXT, DOCX)
- Chunk documents into smaller segments for efficient processing
- Generate embeddings using Ollama LLM
- Retrieve answers to user questions using LLM-based similarity search
- Track question-answer history
- Dockerized for easy deployment

## Project Structure

├── frontend.py # Contains Streamlit code for user interface and interactions 
├── backend.py # Contains the logic for document loading, chunking, embedding creation, and question-answering ├
── Dockerfile # Dockerfile to create a container for the application 
├── requirements.txt # Python dependencies for the project
├── README.md # Documentation for the project


## How It Works

1. **Document Upload**: Users can upload their document through the Streamlit interface.
2. **Document Processing**: The uploaded document is split into smaller chunks for efficient retrieval using LangChain's `RecursiveCharacterTextSplitter`.
3. **Embeddings Generation**: Ollama embeddings are created for each document chunk, and these embeddings are stored in a Chroma vector store.
4. **Question-Answering**: When a user asks a question, the system retrieves the most relevant chunks using similarity search and generates an answer based on the retrieved content.

## Prerequisites

- Docker
- Python 3.11+
- Streamlit
- LangChain
- Ollama model (for embeddings)
- PyPDF2, python-docx for document parsing

## Installation and Setup

### Local Machine Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/llm-qa-application.git
    cd llm-qa-application
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application**:
    ```bash
    streamlit run frontend.py
    ```

### Docker Deployment

1. **Build Docker Image**:
    ```bash
    docker build -t llm-qa-app .
    ```

2. **Run Docker Container**:
    ```bash
    docker run -p 8501:8501 llm-qa-app
    ```

3. Access the app in your browser at `http://localhost:8501`.

## Usage

1. Upload your document via the sidebar.
2. Adjust the chunk size and the number of top matches (`k`) if needed.
3. Ask any question based on the document's content.
4. View the generated answer and chat history on the main page.

## Customization

- You can adjust chunk sizes, overlap, or the Ollama model in the `backend.py` file.
- Modify the UI or logic by updating `frontend.py`.

## Future Improvements
- Add support for additional document formats.
- Incorporate more advanced text splitting and retrieval methods.
- Enhance the UI for a better user experience.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to fork this repository and contribute to further development!

