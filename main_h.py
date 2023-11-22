"""Python file to serve as the frontend"""
import streamlit as st
import faiss
import pickle
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from dotenv import load_dotenv
load_dotenv()
# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index

def get_conversation_chain(vetorestore):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    template = """
    You are an AI assistant for answering questions about the HAI Notion.
    Provide a conversational answer in korean.
    If you don't know the answer, just say '잘 모르겠습니다. ... 😔. 
    Don't try to make up an answer.
    If the question is not about the HAI Notion, politely inform them that you are tuned to only answer questions about HAI Notion.
    
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )
    conversation_chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=QA_CHAIN_PROMPT)
    return conversation_chain
    

icon = "hai.png"
st.set_page_config(page_title="HAI Notion QA Bot", page_icon=icon)
st.title("Notion QA Chatbot :blue[HAI] ")
# From here down is all the StreamLit UI.
if "conversation" not in st.session_state:
    st.session_state.conversation = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

if "processComplete" not in st.session_state:
    st.session_state.processComplete = None

st.session_state.conversation = get_conversation_chain(store) 
st.session_state.processComplete = True

if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", 
                                    "content": "안녕하세요! 궁금하신 것이 있으면 언제든 물어봐주세요!"}]
# Display chat messages from history on app rerun
# Custom avatar for the assistant, default avatar for user
for message in st.session_state.messages:
    if message["role"] == 'assistant':
        with st.chat_message(message["role"], avatar=icon):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
history = StreamlitChatMessageHistory(key="chat_messages")
# Chat logic
if query := st.chat_input("질문을 입력해주세요."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant", avatar=icon):
        chain = st.session_state.conversation
        # Simulate stream of response with milliseconds delay
        with st.spinner("Thinking..."):
            result = chain({"question": query})
            with get_openai_callback() as cb:
                st.session_state.chat_history = result['chat_history']
            response = result['answer']
            source_documents = result['source_documents']
            st.markdown(response)
            with st.expander("참고 문서 확인"):
                st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
    st.session_state.messages.append({"role": "assistant", "content": response})


