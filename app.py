import fitz  # PyMuPDF
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import chainlit as cl

# --- LLM CONFIGURATION ---

# llm_local = ChatOllama(model="mistral:instruct")

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.6, # Slightly lowered for more factual responses
    convert_system_message_to_human=True
)

# --- PROMPT TEMPLATE ---
custom_prompt_template = """You are an expert assistant. Use ONLY the following pieces of context to answer the question at the end.
If you don't know the answer from the provided context, just say that you don't know. DO NOT try to make up an answer.
Keep the answer concise and relevant to the document.

Context: {context}
Question: {question}

Helpful Answer:"""

CUSTOM_PROMPT = PromptTemplate(
    template=custom_prompt_template, input_variables=["context", "question"]
)

# --- CHAINLIT APP ---

@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=100,
            timeout=180,
        ).send()

    file = files[0]
    
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    try:
        # Open the PDF with fitz
        with fitz.open(file.path) as doc:
            if doc.is_encrypted:
                await cl.Message(content=f"Error: The file `{file.name}` is password-protected. Please upload a decrypted file.").send()
                return

            # Text and Metadata Extraction
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = []
            metadatas = []
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text: # Process only if the page contains text
                    page_chunks = text_splitter.split_text(page_text)
                    texts.extend(page_chunks)
                    # Add page number to metadata for each chunk
                    metadatas.extend([{"source": file.name, "page_number": page_num + 1}] * len(page_chunks))

        # Check if any text was extracted
        if not texts:
            await cl.Message(content=f"Error: Could not extract any text from `{file.name}`. The file might be empty or contain only images.").send()
            return

        # Create Chroma vector store
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        docsearch = await cl.make_async(Chroma.from_texts)(
            texts, embeddings, metadatas=metadatas
        )
        
        # Initialize memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
        )

        # Create the retrieval chain with the custom prompt
        retriever = docsearch.as_retriever(search_kwargs={'k': 4})
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm_gemini,
            chain_type="stuff",
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT}
        )

        msg.content = f"Processing `{file.name}` done. You can now ask questions!"
        await msg.update()
        
        cl.user_session.set("chain", chain)

    except Exception as e:
        # Generic error handler
        await cl.Message(content=f"An unexpected error occurred while processing the file: {str(e)}").send()
        return

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler()
    
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]

    text_elements = [] 
    
    if source_documents:
        # Use a set to avoid duplicate source names
        unique_source_names = set()
        for source_doc in source_documents:
            page_num = source_doc.metadata.get("page_number", "N/A")
            source_name = f"Page {page_num}"
            if source_name not in unique_source_names:
                unique_source_names.add(source_name)
                text_elements.append(
                    cl.Text(content=source_doc.page_content, name=source_name)
                )
        
        source_names = sorted(list(unique_source_names))
        
        if source_names:
            answer += f"\n\n**Sources:** {', '.join(source_names)}"
        else:
            answer += "\n\nNo sources found."
    
    await cl.Message(content=answer, elements=text_elements).send()