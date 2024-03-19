from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

prompt = ChatPromptTemplate(
    input_variables=["content"],
    messages=[
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

while True:
    content = input(">> ")

    print(f"You entered: {content}")
