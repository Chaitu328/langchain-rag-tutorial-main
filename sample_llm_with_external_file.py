import os
from dotenv import load_dotenv
from langchain.agents import AgentType
from langchain.agents import load_tools 
from langchain.agents import initialize_agent
from langchain_openai import ChatOpenAI
from langchain.llms import OpenAI

load_dotenv()

serpapi_api_key2 = os.environ["serpapi_api_key"]
llm=OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))


tool=load_tools(["serpapi"],serpapi_api_key=serpapi_api_key2,llm=llm)

agent=initialize_agent(tool,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)

agent.invoke("can you tell me 5 top current affairs?")