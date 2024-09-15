from tool.internet_search_tool import InternetSearchTool
from crewai import Agent, Crew, Process, Task
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import yaml
from dotenv import load_dotenv
load_dotenv()
import streamlit as st

# Set enviornment
google_api_key = os.environ['GOOGLE_API_KEY']
llm = ChatGoogleGenerativeAI(
        model="gemini-pro", verbose=True, temperature=0.9, google_api_key=google_api_key
    )


def load_config(config_file_path):
    with open(config_file_path, 'r') as file:
        return yaml.safe_load(file)

agents_config_path = './config/agents.yaml'
tasks_config_path = './config/tasks.yaml'

agents_config = load_config(agents_config_path)
tasks_config = load_config(tasks_config_path)

# Define agents
print(agents_config['research_analyst'])
researcher_analyst = Agent(
    config=agents_config['research_analyst'],
    verbose=True,
    llm= llm,
    memory= True,
    tools=[InternetSearchTool.internet_search_tool],
    allow_delegation=True
)

writer = Agent(
    config=agents_config['writer'],
    verbose=True,
    llm= llm,
    memory= True,
    tools=[InternetSearchTool.internet_search_tool],
    allow_delegation=False
)


# Define tasks
research_task = Task(
    config=tasks_config['research_task'],
    tools=[InternetSearchTool.internet_search_tool],
    agent=researcher_analyst
)

writing_task = Task(
    config=tasks_config['research_task'],
    tools=[InternetSearchTool.internet_search_tool],
    agent=writer
)

# Create and kick off the crew
crew = Crew(
    agents=[researcher_analyst, writer],
    tasks=[research_task, writing_task],
    process= Process.hierarchical,
    full_output=True,
    share_crew=False,
    verbose = True,
    #max_iter=2,
    manager_llm=llm
)

result = crew.kickoff(inputs={'topic': 'compare renewable energy'})
print(result)

"""
with st.sidebar:
    st.header('Renewable Engergy Research')

    with st.form(key='research_form'):
        topic = st.text_input("search")
        submit_button = st.form_submit_button(label = "Run Research")
if submit_button:
    if not topic:
        st.error("Please fill the ticket field")
    else:
        result= crew.kickoff(inputs={'topic': topic})

        st.subheader("Results of research:")
        st.write(result)

st.stop()
"""