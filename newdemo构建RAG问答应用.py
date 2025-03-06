import os
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma  # 修改这里，导入Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories  import ChatMessageHistory
# 设置环境变量
os.environ[
    "OPENAI_API_KEY"] = ''
os.environ["http_proxy"] = "127.0.0.1:7890"
os.environ["https_proxy"] = "127.0.0.1:7890"

try:
    # 使用WebBaseLoader加载网页内容
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from web")

    # 初始化文本分割器
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # 分割文档
    splits = splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks")

    # 创建嵌入模型
    embeddings = OpenAIEmbeddings()

    # 创建Chroma向量存储
    # 可以指定一个持久化路径以便重用
    persist_directory = "./chroma_db"
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory  # 可选，用于持久化存储
    )

    # 如果需要持久化保存
    # vectorstore.persist()

    # 创建检索器
    retriever = vectorstore.as_retriever()

    # 初始化ChatOpenAI
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # 创建提示模板
    system_prompt = """你是一个专业的AI助手，专注于回答关于AI Agent的问题。
    使用以下检索到的上下文信息来回答问题。
    如果你不知道答案，请直接说不知道，不要编造信息。
    尽量提供全面、准确和具有洞察力的回答。

    检索到的上下文:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # 创建文档链
    document_chain = create_stuff_documents_chain(llm, prompt)
    contexttualize_q_system_prompt="""Given a chat-history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. DO NOT answer the question, just reformulate it if needed and otherwise return it as is."""
    retriever_history_temp = ChatPromptTemplate.from_messages([
        ("system", contexttualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    history_chain = create_history_aware_retriever(llm,retriever,retriever_history_temp)
    #保持问答的历史记录
    store = {}
    def get_session_history(session_id: str):
        if session_id  not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    #创建一个父chain
    chain = create_retrieval_chain(history_chain,document_chain)
    result_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer',
    )
    #第一轮对话
    resp1 = result_chain.invoke(
        {"input":'What is task decomposition?'},
        config={'configurable': {'session_id':'zs123456'}}
    )
    print(resp1['answer'])
    #第二轮对话
    resp2 = result_chain.invoke(
        {"input":'What are common ways of doing it'},
        config={'configurable': {'session_id':'zs123456'}}
    )
    print(resp2['answer'])

except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")