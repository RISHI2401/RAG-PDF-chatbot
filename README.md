# 📄 Conversational PDF Chatbot with Gemini & LangChain

This project is a powerful, conversational AI chatbot that can answer questions about any PDF document you upload. It's built using a RAG (Retrieval-Augmented Generation) architecture, ensuring that the answers are grounded in the document's content.

![alt text](image-1.png)

---

## ✨ Features

- **Chat with any PDF:** Upload a PDF and start asking questions immediately.
- **Conversational Memory:** The chatbot remembers previous parts of the conversation for follow-up questions.
- **Source-Cited Answers:** Responses include references to the page number(s) in the PDF where the information was found.
- **Flexible LLM Backend:** Easily switch between powerful cloud models like Google Gemini and local models running via Ollama.
- **Interactive UI:** A clean and user-friendly interface built with Chainlit.

---

## 🛠️ Tech Stack

- **Orchestration:** LangChain
- **LLMs:** Google Gemini (`gemini-1.5-flash`), Ollama (`mistral:instruct`)
- **Embeddings:** Ollama (`nomic-embed-text`)
- **Vector Store:** ChromaDB
- **UI:** Chainlit
- **PDF Parsing:** PyMuPDF

---

## ⚙️ Setup and Installation

Follow these steps to get the project running on your local machine.

### **1. Clone the Repository**

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

### **2. Create a Virtual Environment**

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Set Up Local Models (Ollama)**

This project relies on Ollama to serve the embedding model and optionally the local LLM.

1.  [Download and install Ollama](https://ollama.com/).
2.  Pull the required models from the command line:
    ```bash
    ollama pull nomic-embed-text
    ollama pull mistral:instruct
    ```
3.  Ensure the Ollama application is running in the background.

### **5. Set Up API Keys**

To use Google Gemini, you need an API key.

1.  Create a file named `.env` in the root of the project folder.
2.  Add your API key to this file:
    ```env
    GEMINI_API_KEY="your_google_api_key_here"
    ```
    _The `.gitignore` file is configured to prevent this file from being uploaded to GitHub._

---

## ▶️ How to Run

With your environment activated and Ollama running, start the Chainlit app:

```bash
chainlit run app.py -w
```

- The `-w` flag enables auto-reloading, so the app will update whenever you save changes to the code.
- Open your browser to `http://localhost:8000` to use the application.
