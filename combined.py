import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_history_aware_retriever
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain import hub
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import create_openai_tools_agent, AgentExecutor,initialize_agent,AgentType

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['HUGGING_API_KEY'] = os.getenv("HUGGING_API_KEY")

# Initialize LLM and Embeddings
llm = ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192",streaming=True)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Streamlit session states
if "feature" not in st.session_state:
    st.session_state.feature = "Search Engine"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
    

# Define Streamlit app
st.set_page_config(page_title="Document ChatBot", page_icon=":robot:", layout="wide")
st.title("Intelligent Assistant with Chat and Retrieval")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assisstant","content":""}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])
    
st.sidebar.title("Choose a Feature")
feature = st.sidebar.radio("Select Feature", ["Search Engine", "Document Upload"])

st.session_state.feature = feature

# Wikipedia and Arxiv Tools
wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
search = DuckDuckGoSearchRun(name = 'Search')
tools = [wiki, arxiv]

from langchain_core.prompts import ChatPromptTemplate

# Define a corrected prompt to meet `create_openai_tools_agent` requirements
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.prompt import PromptTemplate

# Define a simplified prompt template
search_prompt = hub.pull("hwchase17/openai-functions-agent")

agent = create_openai_tools_agent(llm,tools,search_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,return_intermediate_steps=True)

# Search Engine Feature
if st.session_state.feature == "Search Engine":
    st.write("### Chat with Search Engine")
    st.write("Currently using: Wikipedia and  Arxiv")

    # Placeholder for logs
    logs_placeholder = st.empty()

    # Text input for the user's query
    search_query = st.text_input("Enter your search query:")

    if st.button("Search"):
        if search_query.strip():  # Check if query is not empty
            try:
                # Initialize placeholders for logs and response
                logs_placeholder = st.empty()
                response_placeholder = st.empty()

                # Log simulation: Initialize logs list
                logs = []

                # Start logging the execution chain
                logs.append("> Entering new AgentExecutor chain...\n")
                logs_placeholder.code("\n".join(logs), language="text")

                # Run the agent and capture intermediate steps
                response = agent_executor.invoke({"input": search_query})

                # Process and log intermediate steps
                intermediate_steps = response.get("intermediate_steps", [])
                for step in intermediate_steps:
                    if isinstance(step, tuple) and len(step) == 2:
                        action, result = step  # Unpack the ToolAgentAction and response tuple

                        # Extract tool details
                        tool_name = action.tool if hasattr(action, 'tool') else "Unknown Tool"
                        tool_input = action.tool_input if hasattr(action, 'tool_input') else {}
                        tool_log = action.log if hasattr(action, 'log') else "No log available"

                        # Log the tool invocation
                        logs.append(f"{tool_log.strip()}\n")
                        logs.append(f"Response from `{tool_name}`: {result.strip() if isinstance(result, str) else result}\n")
                    else:
                        logs.append(f"Unexpected step format: {step}\n")

                logs.append("> Finished chain.\n")
                logs_placeholder.code("\n".join(logs), language="text")

                # Display assistant's response dynamically
                output = response.get("output", "No response found.")
                response_placeholder.markdown(f"**Response:** {output}")

                # Append to chat history
                st.session_state.messages.append({"role": "assistant", "content": output})

            except Exception as e:
                # Handle and log errors
                st.error(f"An error occurred: {e}")
                logs.append(f"Error: {str(e)}")
                logs_placeholder.code("\n".join(logs), language="text")

# Document Upload Feature
if st.session_state.feature == "Document Upload":
    st.write("### Upload and Process Documents")
    option = st.selectbox("Select Document Type", [".txt File", "PDF File", "Web Page"])
    uploaded_file = st.file_uploader("Upload your document", type=["txt", "pdf"]) if option != "Web Page" else None
    web_page_url = st.text_input("Enter Web Page URL") if option == "Web Page" else None

    if st.button("Process Document"):
        if option == ".txt File" and uploaded_file is not None:
            with open("temp_text_file.txt", "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = TextLoader("temp_text_file.txt")
            text_documents = loader.load()

        elif option == "PDF File" and uploaded_file is not None:
            with open("temp_pdf_file.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader("temp_pdf_file.pdf")
            text_documents = loader.load()

        elif option == "Web Page" and web_page_url:
            loader = WebBaseLoader(web_page_url)
            text_documents = loader.load()

        else:
            st.error("Please upload a valid document or provide a web page URL.")
            st.stop()

        # Load and split the document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(text_documents)

        # Create vector store
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        st.session_state.retriever = vectorstore.as_retriever()

        # Define system prompts
        system_prompt = (
            "You are an assistant specialized in answering questions about the uploaded document. "
            "Provide detailed and structured answers. If technical terms are mentioned, explain them clearly.\n\n{context}"
        )

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "provide an elaborate answer according to the question "
            "asked, explaining technical terms if necessary."
        )

        # Create history-aware retriever prompt
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(llm, st.session_state.retriever, contextualize_q_prompt)

        # Define prompts
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        st.session_state.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        st.success("Document processed successfully!")

    # Chat Section
    st.header("Chat with the Document")
    user_input = st.text_input("Ask a question about the document:")

    if user_input:
        if st.session_state.rag_chain is None:
            st.error("Please process a document first using the 'Process Document' button.")
        else:
            try:
                response = st.session_state.rag_chain.invoke(
                    {"input": user_input, "chat_history": st.session_state.chat_history}
                )
                st.session_state.chat_history.extend([
                    HumanMessage(content=user_input),
                    AIMessage(content=response["answer"])
                ])
                st.success(f"Answer: {response['answer']}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Clear Chat History Button
if st.button("Clear Chat History"):
    st.session_state.chat_history.clear()
    st.write("Chat history cleared.")