import streamlit as st
import os
import tempfile
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import warnings
warnings.filterwarnings("ignore")

class DocumentQAAssistant:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.embeddings = None
        
    def initialize_embeddings(self, use_openai=True, openai_api_key=None):
        """Initialize embeddings model"""
        if use_openai and openai_api_key:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=openai_api_key,
                model="text-embedding-ada-002"
            )
        else:
            # Use free local embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
    
    def load_documents(self, uploaded_files):
        """Load and process uploaded documents"""
        all_docs = []
        
        for uploaded_file in uploaded_files:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Load document based on file type
                if uploaded_file.name.lower().endswith('.pdf'):
                    loader = PyPDFLoader(tmp_file_path)
                elif uploaded_file.name.lower().endswith('.docx'):
                    loader = Docx2txtLoader(tmp_file_path)
                else:  # Assume text file
                    loader = TextLoader(tmp_file_path, encoding='utf-8')
                
                documents = loader.load()
                all_docs.extend(documents)
                st.success(f"Loaded {len(documents)} pages from {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {str(e)}")
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        return all_docs
    
    def process_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        st.info(f"Split into {len(chunks)} chunks")
        return chunks
    
    def create_vector_store(self, chunks, persist_directory="./chroma_db"):
        """Create vector store from document chunks"""
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        return self.vector_store
    
    def setup_qa_chain(self, use_openai=True, openai_api_key=None, model_name="gpt-3.5-turbo"):
        """Set up the QA chain"""
        if use_openai and openai_api_key:
            llm = ChatOpenAI(
                openai_api_key=openai_api_key,
                model_name=model_name,
                temperature=0.1
            )
        else:
            # For local models, you would use something like:
            # from langchain.llms import LlamaCpp
            # llm = LlamaCpp(model_path="path/to/your/model")
            st.error("OpenAI API key required for this version. See advanced version for local models.")
            return None
        
        # Custom prompt for better answers
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {question}
        Answer: """
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return self.qa_chain
    
    def query_documents(self, question):
        """Query the document collection"""
        if not self.qa_chain:
            return "Please process documents first and set up the QA chain."
        
        try:
            result = self.qa_chain({"query": question})
            return result
        except Exception as e:
            return f"Error during query: {str(e)}"

def main():
    st.set_page_config(
        page_title="AI Document Query Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 AI-Powered Document Query Assistant")
    st.markdown("Upload your documents and ask questions about their content!")
    
    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = DocumentQAAssistant()
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API key input
        use_openai = st.radio("Choose model type:", ["OpenAI GPT", "Local Model"])
        openai_api_key = None
        
        if use_openai == "OpenAI GPT":
            openai_api_key = st.text_input("OpenAI API Key:", type="password")
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
            model_name = st.selectbox(
                "Select model:",
                ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"]
            )
        else:
            st.info("Local model support requires additional setup with Ollama or similar.")
            model_name = "local"
        
        # Document upload
        st.header("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or DOCX files"
        )
        
        # Processing parameters
        st.header("🔧 Processing Options")
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)
        
        # Process button
        if st.button("🚀 Process Documents", use_container_width=True):
            if not uploaded_files:
                st.error("Please upload at least one document.")
            elif use_openai == "OpenAI GPT" and not openai_api_key:
                st.error("Please enter your OpenAI API key.")
            else:
                with st.spinner("Processing documents..."):
                    try:
                        # Initialize embeddings
                        st.session_state.assistant.initialize_embeddings(
                            use_openai=(use_openai == "OpenAI GPT"),
                            openai_api_key=openai_api_key
                        )
                        
                        # Load documents
                        documents = st.session_state.assistant.load_documents(uploaded_files)
                        
                        if documents:
                            # Process documents
                            chunks = st.session_state.assistant.process_documents(
                                documents, chunk_size, chunk_overlap
                            )
                            
                            # Create vector store
                            st.session_state.assistant.create_vector_store(chunks)
                            
                            # Setup QA chain
                            qa_chain = st.session_state.assistant.setup_qa_chain(
                                use_openai=(use_openai == "OpenAI GPT"),
                                openai_api_key=openai_api_key,
                                model_name=model_name
                            )
                            
                            if qa_chain:
                                st.session_state.processed = True
                                st.success("✅ Documents processed successfully!")
                                st.balloons()
                        
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
    
    # Main chat area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat with Your Documents")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Source Documents"):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"**Source {i+1}:** {source.page_content[:200]}...")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            if not st.session_state.processed:
                st.error("Please process documents first in the sidebar.")
            else:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        result = st.session_state.assistant.query_documents(prompt)
                        
                        if isinstance(result, dict) and "result" in result:
                            response = result["result"]
                            source_docs = result.get("source_documents", [])
                            
                            st.markdown(response)
                            
                            # Display sources if available
                            if source_docs:
                                with st.expander("📚 Source Documents"):
                                    for i, doc in enumerate(source_docs):
                                        source_text = doc.page_content.replace('\n', ' ')[:300]
                                        st.markdown(f"**Source {i+1}:** {source_text}...")
                                        st.markdown(f"*Metadata:* {doc.metadata.get('source', 'Unknown')}")
                                        st.markdown("---")
                            
                            # Add assistant response to chat history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response,
                                "sources": source_docs
                            })
                        else:
                            st.error("Failed to get response from the model.")
    
    with col2:
        st.header("📊 System Info")
        
        if st.session_state.processed:
            st.success("✅ System Ready")
            st.metric("Status", "Active")
            
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
            
            if st.button("Reset System"):
                st.session_state.assistant = DocumentQAAssistant()
                st.session_state.processed = False
                st.session_state.messages = []
                st.rerun()
        else:
            st.warning("⚠️ System Not Ready")
            st.metric("Status", "Inactive")
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Upload multiple related documents
        - Ask specific, focused questions
        - Use chunk size 800-1200 for most documents
        - Check source documents for verification
        """)

if __name__ == "__main__":
    main()
