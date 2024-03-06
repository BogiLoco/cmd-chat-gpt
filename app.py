from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory

import warnings
warnings.filterwarnings("ignore")

# Model object
llm = ChatOpenAI(temperature=1)

# Stores the history of the chat
history = FileChatMessageHistory('chat_history.json')

# Memory
memory = ConversationBufferMemory(
    memory_key='chat_history',
    chat_memory=history,
    return_messages=True
)

# Prompt template
prompt = ChatPromptTemplate(
    input_variables=["content"],
    messages=[
        SystemMessage(content='You are chatbot having conversation with a human.'),
        MessagesPlaceholder(variable_name='chat_history'), 
        HumanMessagePromptTemplate.from_template('{content}') 
    ]
)

# Chain object
chain = LLMChain(
    llm=llm, 
    prompt=prompt,
    memory=memory,
    verbose = False
)

if __name__ == "__main__":
    
    print('-' * 50)
    print('Started application')
    while True:
        content = input('Your input: ')
        if content in ['quit', 'exit', 'bye']:
            print('Goodbye')
            exit(0)
        
        response = chain.run({'content': content})
        print(response)
        print('-' * 50)