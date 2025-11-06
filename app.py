# app.py
# Updated: improved error handling, fallback LLM selection (Gemini -> Ollama),
# simpler file ask flow, optional persistent Chroma store, and additional checks.
# Plan:
# 1. On chat start, prompt user to upload a PDF (single await, no busy-wait).
# 2. Extract text via PyMuPDF (fitz), chunk it, create or load a Chroma store.
# 3. Build a ConversationalRetrievalChain using Gemini if API key present,
#    otherwise fallback to local ChatOllama (if available).
# 4. Persist the chain into user_session for subsequent messages.
# 5. On each message, invoke chain, return answer + source page texts.

import os
import logging
import fitz  # PyMuPDF
import chainlit as cl

# --- LangChain / models / embeddings imports ---
# Updated imports for modern LangChain 1.x
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# Chains and Memory are now in langchain_classic in LangChain 1.x
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

# Community / third-party connectors
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Prompt template (kept concise, still enforced to rely only on context) ---
custom_prompt_template = """You are an expert assistant. Use ONLY the following pieces of context to answer the question at the end.
If you don't know the answer from the provided context, just say that you don't know. DO NOT try to make up an answer.
Keep the answer concise and relevant to the document.

Context: {context}
Question: {question}

Helpful Answer:"""

CUSTOM_PROMPT = PromptTemplate(
    template=custom_prompt_template, input_variables=["context", "question"]
)

# --- LLM selection logic ---
# Use Google Gemini if GEMINI_API_KEY is set; otherwise fall back to a local ChatOllama.
def get_llm():
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        logger.info("Using Google Gemini (cloud) LLM")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",  # Updated to use -latest suffix for current API
            google_api_key=gemini_key,
            temperature=0.6,
            convert_system_message_to_human=True
        )
    else:
        # Fallback to local model via Ollama (ensure Ollama is running locally)
        logger.info("GEMINI_API_KEY not set — falling back to local Ollama model (mistral:instruct)")
        try:
            return ChatOllama(model="mistral:instruct")
        except Exception as e:
            # If even the fallback fails, raise with helpful message
            raise RuntimeError(
                "No LLM available: GEMINI_API_KEY not set and ChatOllama failed to initialize. "
                "Either set GEMINI_API_KEY or make sure Ollama is installed and running."
            ) from e

# --- Configuration for Chroma persistence (optional) ---
# To avoid recomputing embeddings for the same PDF repeatedly, we can persist Chroma
# to a local directory. Set CHROMA_PERSIST_DIR to enable persistence.
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_store")

# --- Chainlit handlers ---

@cl.on_chat_start
async def on_chat_start():
    """
    When a chat starts, ask the user for a PDF (single await). Process the PDF:
    - reject encrypted PDFs,
    - extract text per page,
    - chunk text,
    - create embeddings and a Chroma vector store (persisted optionally),
    - create a ConversationalRetrievalChain and save it to the user session.
    """
    # Prompt for file upload (Chainlit will wait for upload or timeout)
    files = await cl.AskFileMessage(
        content="Please upload a PDF file to begin (scanned PDFs may not extract text).",
        accept=["application/pdf"],
        max_size_mb=200,
        timeout=180
    ).send()

    if not files:
        await cl.Message(content="No file uploaded. Please try again.").send()
        return

    file = files[0]
    processing_msg = cl.Message(content=f"Processing `{file.name}`...")
    await processing_msg.send()

    try:
        # Open the PDF with PyMuPDF
        with fitz.open(file.path) as doc:
            if doc.is_encrypted:
                await cl.Message(content=f"Error: The file `{file.name}` is password-protected. Please upload a decrypted file.").send()
                return

            # Extract text from pages
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = []
            metadatas = []
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text and page_text.strip():
                    page_chunks = text_splitter.split_text(page_text)
                    texts.extend(page_chunks)
                    # Save metadata with page number and filename
                    metadatas.extend([{"source": file.name, "page_number": page_num + 1}] * len(page_chunks))

        if not texts:
            await cl.Message(content=f"Error: Could not extract any text from `{file.name}`. The file might be scanned images only. Consider OCRing it and uploading text/PDF with selectable text.").send()
            return

        # The embeddings model: OllamaEmbeddings (requires Ollama running and 'nomic-embed-text' model available)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Create or load a Chroma vectorstore — use persist directory to save embeddings between runs
        # In modern Chroma, persistence is automatic when persist_directory is provided
        persist_path = os.path.join(CHROMA_PERSIST_DIR, file.name.replace(" ", "_").replace(".", "_"))
        docsearch = Chroma.from_texts(
            texts,
            embeddings,
            metadatas=metadatas,
            persist_directory=persist_path
        )
        # Note: persist() is deprecated in newer Chroma versions - persistence is automatic

        # Setup conversational memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
        )

        retriever = docsearch.as_retriever(search_kwargs={"k": 4})

        # Build LLM (Gemini if available, else Ollama)
        llm = get_llm()

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT}
        )

        # Update the processing message - in Chainlit 2.6+, we need to remove and send a new message
        await processing_msg.remove()
        await cl.Message(content=f"Processing `{file.name}` done. You can now ask questions!").send()
        # Store chain in user session
        cl.user_session.set("chain", chain)

    except Exception as e:
        logger.exception("Unexpected error processing PDF")
        await cl.Message(content=f"An unexpected error occurred while processing the file: {str(e)}").send()
        return


@cl.on_message
async def main(message: cl.Message):
    """
    Handles each chat message: invokes the saved chain and returns the answer.
    Also attaches text elements for the source pages referenced.
    """
    chain = cl.user_session.get("chain")
    if not chain:
        await cl.Message(content="No document loaded. Please restart and upload a PDF.").send()
        return

    # Note: AsyncLangchainCallbackHandler removed due to incompatibility with LangChain 1.x
    # The callback was optional and used for streaming/progress updates
    try:
        # Invoke chain (async) - updated to use proper input format
        res = await chain.ainvoke({"question": message.content})
    except AttributeError:
        # If ainvoke does not exist, try invoke (sync) via an async wrapper
        res = await cl.make_async(chain.invoke)({"question": message.content})
    except Exception as e:
        logger.exception("Error invoking chain")
        await cl.Message(content=f"An error occurred: {str(e)}").send()
        return

    answer = res.get("answer") or res.get("output_text") or "No answer returned."
    source_documents = res.get("source_documents", [])

    text_elements = []
    unique_pages = set()
    for src in source_documents:
        page_num = src.metadata.get("page_number", "N/A")
        source_name = f"Page {page_num}"
        if source_name not in unique_pages:
            unique_pages.add(source_name)
            # Attach the chunk text with the page name
            text_elements.append(cl.Text(content=src.page_content, name=source_name))

    # Append a small footer listing page sources (if any)
    if unique_pages:
        sources_list = ", ".join(sorted(unique_pages, key=lambda s: int(s.split(" ")[1]) if s.split(" ")[1].isdigit() else s))
        answer += f"\n\n**Sources:** {sources_list}"
    else:
        answer += "\n\nNo sources found."

    await cl.Message(content=answer, elements=text_elements).send()
