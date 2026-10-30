from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage,HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages


from typing import TypedDict,Annotated,Literal
from pydantic import BaseModel, Field
import operator
from dotenv import load_dotenv

import os
load_dotenv()  # Load environment variables from .env file

model = ChatOpenAI(
    model="openai/gpt-oss-120b",
    temperature=0,
    openai_api_key=os.getenv("GROQ_API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1",
    timeout=60,
    max_retries=5,
    max_tokens=4096,
    reasoning_effort="low",
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
def chat_node(state:ChatState) -> dict:
    #take user query from the state
    messages = state['messages']
    # call the model with the messages
    response = model.invoke(messages)
    #response store in the state
    return {'messages':[response]}

checkpoint=MemorySaver()
graph=StateGraph(ChatState)
#add nodes
graph.add_node('chat_node',chat_node)
#add edges
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot=graph.compile(checkpointer=checkpoint)


config = {"configurable": {"thread_id": "1"}}

initial_state = {
    'messages': [HumanMessage(content="what is my name?")]
}
response = chatbot.invoke(initial_state, config=config)
response['messages'][-1].content


thread_id="1"
while True:
    user_message = input("Type your message: ")
    print("User:", user_message)
    if user_message.strip().lower() in ["exit", "quit",'bye']:
        print("Exiting the chatbot.")
        break
    config={'configurable': {'thread_id': thread_id}}
    response = chatbot.invoke({'messages': [HumanMessage(content=user_message)]}, config=config)
    print('AI:', response['messages'][-1].content)