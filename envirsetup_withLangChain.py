import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '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'
os.environ['USER_AGENT'] = 'my_custom_user_agent'
os.environ['OPENAI_API_KEY'] = 'sk-proj-wUDYbhvDsdZWb0KVUuzJT3BlbkFJY1r4rGhwj8qUgSKBvokk'


import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from operator import itemgetter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


#### INDEXING ####

## Load Docs
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()
# Print docs
print(len(docs[0].page_content))
print(docs[0].page_content[:500])


## Split
text_splitter = RecursiveCharacterTextSplitter( ####改
    chunk_size=1000, 
    chunk_overlap=200,
    add_start_index=True
)
splits = text_splitter.split_documents(docs)
# Print
print(f"Number of splits: {len(splits)}")
print(splits[0].page_content)


## Embed
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings()
)
# Create
retriever = vectorstore.as_retriever()

#### RETRIEVAL and GENERATION ####

# prompt template
prompt = hub.pull("rlm/rag-prompt")

# creat prompt
# prompt = HumanMessage(
#     content=augment_prompt(query)
# )

# messages.append(prompt)

# res = chat(messages)

# print(res.content)

# Initialize the language model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Function to format retrieved documents for LLM input
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the RAG chain
rag_chain = (
    {"context": retriever | format_docs, 
    "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Invoke the chain with a sample question
result = rag_chain.invoke("What is Task Decomposition?")
len(result)
print(result)


# # Prompt

# generate_queries = (
#     prompt_perspectives 
#     | ChatOpenAI(temperature=0) 
#     | StrOutputParser() 
#     | (lambda x: x.split("\n"))
# )
# from langchain.load import dumps, loads

# def get_unique_union(documents: list[list]):
#     """ Unique union of retrieved docs """
#     flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
#     unique_docs = list(set(flattened_docs))    # Get unique docs
#     return [loads(doc) for doc in unique_docs]

# # Retrieve
# question = "What is task decomposition for LLM agents?"
# retrieval_chain = generate_queries | retriever.map() | get_unique_union
# docs = retrieval_chain.invoke({"question":question})
# len(docs)


# # RAG
# template = """Answer the following question based on this context:
# {context}
# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)
# llm = ChatOpenAI(temperature=0)

# final_rag_chain = (
#     {"context": retrieval_chain, 
#      "question": itemgetter("question")} 
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# final_rag_chain.invoke({"question":question})

