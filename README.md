# RAG (Retrieval-Augmented Generation)

Este código implementa un sistema de RAG (Retrieval-Augmented Generation), a base de funciones. El sistema combina capacidades de recuperación de información con generación de texto para responder a consultas, apoyándose en un documento específico como base de conocimiento.

    
    from google.colab import drive
    drive.mount('/content/drive')
    
    import subprocess

#Configurar el entorno e instalar paquetes
    
    try:
        subprocess.run(['pip', 'install', 'langchain', 'langchain_community', 'langchain-openai',
                        'scikit-learn', 'langchain-ollama', 'pymupdf', 'langchain_huggingface',
                        'faiss-gpu'], check=True)
        print("Paquetes instalados correctamente.")
    except subprocess.CalledProcessError as e:
        print("Error al instalar paquetes:", e)

#Cargar y preparar documentos

    from langchain.document_loaders import PyMuPDFLoader
    
    from langchain.text_splitter import RecursiveCharacterTextSplitter

#Ruta al archivo PDF en Google Drive

    file_path = '/content/drive/MyDrive/Colab Notebooks/CIDE/Introducción a la Estadística.pdf'

#Cargar el documento desde el archivo PDF

    loader = PyMuPDFLoader(file_path)
    
    docs = loader.load()

#Se divide el contenido del documento en trozos manejables usando un divisor recursivo.

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=2000, chunk_overlap=200)
    doc_splits = text_splitter.split_documents(docs)

#Configuración del Modelo Ollama. !ollama instala y prepara el modelo localmente.

    !curl -fsSL https://ollama.com/install.sh | sh
    
    !pip install colab-xterm
    
    %load_ext colabxterm
    
    %xterm
    
    #ollama serve
    
    !ollama pull llama3

#Se utiliza FAISS para crear un almacén vectorial a partir de los fragmentos.

    from langchain_huggingface import HuggingFaceEmbeddings

#Crear almacen vectorial

    from langchain_community.vectorstores import FAISS
    
    vectorstore = FAISS.from_documents(
         documents=doc_splits,
         embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
         #embedding=HuggingFaceEmbeddings()
     )
     
    vretriever = vectorstore.as_retriever(k=4)


#Configurar la plantilla LLM y prompt. Se define un modelo de lenguaje y un prompt para guiar la generación de respuestas.

    from langchain_ollama import ChatOllama
    
    from langchain.prompts import PromptTemplate
    
    from langchain_core.output_parsers import StrOutputParser
    
    prompt = PromptTemplate(
        template="""Eres un asistente para tareas de preguntas y respuestas.
    Usa los siguientes documentos para responder la pregunta.
    Si no sabes la respuesta, simplemente di que no lo sabes.
    Utiliza un máximo de tres oraciones y mantén la respuesta concisa:
    Pregunta: {question}
    Documentos: {documents}
    Respuesta:
    """,
        input_variables=["question", "documents"],
    )
    
    llm = ChatOllama(model="llama3", temperature=0)
    
    rag_chain = prompt | llm | StrOutputParser()

#Se define una clase que integra el recuperador de documentos y la cadena RAG.

    class RAGApplication:
        def __init__(self, retriever, rag_chain):
            self.retriever = retriever
            self.rag_chain = rag_chain

        def run(self, question):
            documents = self.retriever.get_relevant_documents(question)
            doc_texts = "\n".join([doc.page_content for doc in documents if hasattr(doc, 'page_content')])
            answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})


#Se inicializa la aplicación y se formula una consulta.

    rag_application = RAGApplication(retriever, rag_chain)
    
    question = "que es un estadistico??"
    answer = rag_application.run(question)
    print("Question:", question)
    print("Answer:", answer)
            return answer´´´
