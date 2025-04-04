import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from IPython.display import Image, display
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="qwen-2.5-32b")

# Define the structured output using Pydantic
class MeetingSummary(BaseModel):
    key_topics: list[str] = Field(description="List of main topics discussed")
    decisions: list[str] = Field(description="List of taken and pending decisions")
    action_items: list[str] = Field(description="Action items with assignees and deadlines")

# Define the structured output parser directly with the Pydantic model
parser = PydanticOutputParser(pydantic_object=MeetingSummary)

class State(TypedDict):
    transcript: str
    summary: str
    action_items: str
    follow_email: str
    messages: Annotated[list, add_messages]

def summarize(state):
    prompt = PromptTemplate.from_template(
        """
        You are an AI assistant that summarizes meeting transcripts.

        Please analyze the transcript and extract:

        - Key topics
        - Decisions made
        - Action items

        {format_instructions}

        Transcript:
        {transcript}
        """
    )
    
    formatted_prompt = prompt.format(
        transcript=state["transcript"],
        format_instructions=parser.get_format_instructions()
    )

    response = llm.invoke(formatted_prompt).content
    structured_data = parser.parse(response)  # This is a Pydantic object
    
    print(f"Structured Data Type: {type(structured_data)}")
    #Log the type of structured_data to debug
    print(f"Structured Data Type: {type(structured_data)}")
    print(f"Structured Data: {structured_data}")

    # If it's already a dict, return it directly, else use .dict()
    if isinstance(structured_data, dict):
        return {"summary": structured_data}
    else:
        return {"summary": structured_data.dict()}


# Building graph
graph = StateGraph(State)

# Adding node
graph.add_node("Summary", summarize)

# Adding edge
graph.add_edge(START, "Summary")
graph.add_edge("Summary", END)

app = graph.compile()

display(Image(app.get_graph().draw_mermaid_png()))

# -------- Streamlit UI --------
st.set_page_config(page_title="Meeting Summarizer", page_icon="📝")
st.title("Meeting Transcript Summarizer")

transcript_input = st.text_area("Paste your meeting transcript below:", height=200)

if st.button("Summarize"):
    if not transcript_input.strip():
        st.warning("Please enter a meeting transcript.")
    else:
        with st.spinner("Generating summary..."):
            result = app.invoke({
                "transcript": transcript_input,
                "summary": None,
                "action_items": None,
                "follow_email": None,
                "messages": [],
            })
        
        # Display structured output
        st.subheader("Meeting Summary")
        if isinstance(result["summary"], dict):
            st.markdown("### Key Topics")
            for topic in result["summary"]["key_topics"]:
                st.write(f"- {topic}")

            st.markdown("### Decisions Made")
            for decision in result["summary"]["decisions"]:
                st.write(f"- {decision}")

            st.markdown("### Action Items")
            for item in result["summary"]["action_items"]:
                st.write(f"- {item}")
