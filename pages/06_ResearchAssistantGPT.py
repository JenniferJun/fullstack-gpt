from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import WikipediaQueryRun
from langchain.tools import DuckDuckGoSearchResults
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import streamlit as st
from typing import Type
from langchain.document_loaders import WebBaseLoader

st.set_page_config(
    page_title="ResearchGPT",
    page_icon="🔍",
)

class WikipediaSearchToolArgsSchema(BaseModel):
    query: str = Field(
        description="The query to research. Example: Research about the XZ backdoor"
    )

class WikipediaSearchTool(BaseTool):
    name = "WikipediaSearchTool"
    description = "Use this to search Wikipedia content for given terms."
    args_schema: Type[WikipediaSearchToolArgsSchema] = WikipediaSearchToolArgsSchema

    def _run(self, query):
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        return wikipedia.run(query)

class DuckDuckGoSearchTool(BaseTool):
    name = "DuckDuckGoSearchTool"
    description = "Use this to return URLs of searched web pages from given terms."
    args_schema: Type[WikipediaSearchToolArgsSchema] = WikipediaSearchToolArgsSchema

    def _run(self, query):
        ddg_search = DuckDuckGoSearchResults()
        return ddg_search.run(query)

class WebContentExtractorToolArgsSchema(BaseModel):
    urls: list = Field(
        description="The urls to extract. urls are list of url."
    )

class WebContentExtractorTool(BaseTool):
    name = "WebContentExtractorTool"
    description = "Extracts web content using WebBaseLoader."

    args_schema: Type[WebContentExtractorToolArgsSchema] = WebContentExtractorToolArgsSchema
    
    def _run(self, urls):
        web_loader = WebBaseLoader(urls)
        loaded_contents = web_loader.load()
        return loaded_contents
    
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

openapi_key = st.sidebar.text_input("OpenAI API KEY : ")

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
    model_name="gpt-4-turbo",
    openai_api_key=openapi_key,
)

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

st.title("Research Assistant GPT")

if not openapi_key:
    st.error("Please enter your OpenAI API key to proceed.")

st.markdown(
    """
 Welcome to Research Assistant GPT.
            
Use this chatbot to research any topic you're interested in!

Ask a question for research and our Assistant will support to research following the steps below.
1. Search Wikipedia for information
2. Search DuckDuckGo for additional sources
3. Extract content from relevant websites
4. Summarize all findings with reference URLs
"""
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if openapi_key:
    tools = load_tools(["ddg-search", "wikipedia"], llm=llm)
    agent = initialize_agent(
        tools=[
            WikipediaSearchTool(),
            DuckDuckGoSearchTool(),
            WebContentExtractorTool(),
        ],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    send_message("I'm ready! What would you like to research?", "ai", save=False)
    paint_history()
    
    message = st.chat_input("What do you want to know? i.e) Research about the XZ backdoor")
    if message:
        send_message(message, "human")
        with st.chat_message("ai"):
            response = agent.run(message)
