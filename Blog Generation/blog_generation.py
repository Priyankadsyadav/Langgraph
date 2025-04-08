#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
from dotenv import load_dotenv
load_dotenv()


# In[2]:


from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

llm=ChatGroq(model="gemma2-9b-it")
llm


# In[ ]:


def generate_title(user_input : str)->str:
    """Generates title based on user's input."""
    response=llm.invoke(f"Generate a blog title for thr topic: {user_input}")
    return response.content


# In[ ]:


def generate_content(title:str)->str:
    """ Content is generated based on title from generate_title function"""

    response = llm.invoke(f"Write a detailed blog post on the topic: {title}")
    return response.content  # Extracting LLM-generated content
    


# In[ ]:


tools=[generate_title,generate_content]
llm_with_tools=llm.bind_tools(tools,parallel_tool_calls=False)


# In[ ]:


from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from typing import Annotated
from langgraph.graph.message import add_messages

class MessagesState(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]


# In[ ]:


from langchain_core.messages import HumanMessage, SystemMessage

# System message
sys_msg = SystemMessage(content="You are a helpful assistant.")


def assistant(state:MessagesState):
    return {"messages":[llm_with_tools.invoke([sys_msg] + state["messages"])]}


# In[ ]:


from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display


# In[ ]:


builder=StateGraph(MessagesState)

#define nodes
builder.add_node("assistant",assistant)
builder.add_node("tools",ToolNode(tools))

#define edges
builder.add_edge(START,"assistant")
builder.add_conditional_edges(
    "assistant",tools_condition
)

builder.add_edge("tools","assistant")

react_graph=builder.compile()


# In[ ]:


# Show
display(Image(react_graph.get_graph().draw_mermaid_png()))


# In[ ]:


messages = [HumanMessage(content="title for expenditure")]
messages = react_graph.invoke({"messages": messages})


# In[ ]:


for m in messages['messages']:
    m.pretty_print()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




