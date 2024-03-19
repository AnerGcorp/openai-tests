from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

# initializing ChatOpenAI
chat = ChatOpenAI()
# adding memory to store the chat history
memory = ConversationBufferMemory(memory_key="messages", return_messages=True)
# chat prompt
prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

# chaining
chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory
)

while True:
    content = input(">> ")

    result = chain.invoke({"content": content})

    print(result["text"])
