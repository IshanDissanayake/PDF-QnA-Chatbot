# PDF Q&A Chatbot
This project is a **Retrieval-Augmented Generation (RAG)** application that enables users to upload PDF files, builds a knowledge base from their contents, and provides intelligent answers to user queries. The application leverages **OpenAI API**, **LangChain framework**, and the **Chroma open-source vector database** for an efficient and robust solution. A combination of semantic search and keyword search strategies is implemented using an **Ensemble Retriever**, ensuring precise retrieval of meaningful text chunks. The application is evaluated using the **Ragas** library and features an intuitive user interface built with **Streamlit**.

## Key Features
* **PDF Document Ingestion**: Upload and index PDF files
* **Hybrid Search Strategy**:
    - Semantic Search: Uses vector embeddings for contextual matching
    - Keyword Search: Enables precise text matching
* **Ensemble Retriever**: Intelligently combines search strategies for optimal results
* **Streamlit User Interface**: Clean, interactive, and user-friendly design
* **RAG Evaluation**: Integrated Ragas library for performance assessment

## Technology Stack
- **Python**: Core language for development.
- **Streamlit**: Used to create the web-based user interface.
- **OpenAI API**: For natural language processing and generation.
- **LangChain Framework**: Provides robust RAG framework integration.
- **Chroma Vector Database**: Efficient storage and retrieval of vector embeddings.
- **Ragas Library**: Evaluates the retrieval performance of the RAG pipeline.

## Installation and Usage
### Installation:
1. Clone this repository: <br>
`git clone https://github.com/IshanDissanayake/PDF-Insight-Assistant.git` <br>
`cd PDF-Insight-Assistant`

2. Create a virtual environment: <br>
  `python -m venv venv` <br>
`venv\Scripts\activate` On Windows <br>
`source venv/bin/activate` On macOS 

4. Install dependencies: <br>
`pip install -r requirements.txt`

5. Set up environment variables for the OpenAI API key: <br>
   `export OPENAI_API_KEY=your_openai_api_key`

### Running the Application
* Start the Streamlit app: <br>
`streamlit run app.py` <br>

Upload your PDF files, ask questions, and explore the intelligent answers generated!

## License
This project is licensed under the `MIT License`.

## Acknowledgments
* `OpenAI` for the GPT API.
* `LangChain` for the RAG framework.
* `Chroma` for the vector database.
* `Streamlit` for the user interface.
* `Ragas` for evaluation metrics.



