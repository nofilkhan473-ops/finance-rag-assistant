import os
import traceback
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import gradio as gr

# Hugging Face will automatically grab your API key from the Settings > Secrets tab!
MY_GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Look for the PDF in the same folder as this script
LOCAL_PDF_PATH = "Gov-AR.pdf"

def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)

def create_embeddings(chunks, index_path="faiss_finance_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        if chunks:
            vectorstore.add_documents(chunks)
            vectorstore.save_local(index_path)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(index_path)
    return vectorstore

# Global variable to hold our vector database
global_vectorstore = None

def respond_to_chat(message, history):
    try:
        if not MY_GROQ_API_KEY:
            return "❌ ERROR: Groq API Key not found. Please add it to the Space Secrets in Settings!"

        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1, api_key=MY_GROQ_API_KEY)
        retriever = global_vectorstore.as_retriever(search_kwargs={"k": 2}) 
        docs = retriever.invoke(message)
        
        if not docs:
            return "I couldn't find any relevant information in the uploaded document."

        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = ChatPromptTemplate.from_template(
            "You are an expert financial analyst. Answer the question based ONLY on the provided context.\n"
            "If the answer is not in the context, state that you do not know.\n\n"
            "Context:\n{context}\n\nQuestion: {question}"
        )
        
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": message})
        
        sources = list(dict.fromkeys([f"Page {doc.metadata.get('page', 0) + 1}" for doc in docs]))
        source_text = "\n\n**Sources:**\n" + "\n".join([f"- PDF Document ({s})" for s in sources])
        
        return response.content + source_text

    except Exception as e:
        print(traceback.format_exc())
        return f"❌ SYSTEM ERROR:\n`{str(e)}`"

# Initialize everything before launching the UI
if os.path.exists(LOCAL_PDF_PATH):
    print("Splitting PDF into chunks...")
    document_chunks = load_and_split_pdf(LOCAL_PDF_PATH)
    
    print("Initializing Embeddings & Vector Database...")
    global_vectorstore = create_embeddings(document_chunks)
else:
    print(f"⚠️ ERROR: Could not find {LOCAL_PDF_PATH} in the Space files.")

# Launch the Gradio App (No debug or share flags needed for HF Spaces)
demo = gr.ChatInterface(
    fn=respond_to_chat,
    title="📊 Finance RAG Assistant",
    description="Ask questions about the State Bank of Pakistan Annual Report."
)

if __name__ == "__main__":
    demo.launch()
