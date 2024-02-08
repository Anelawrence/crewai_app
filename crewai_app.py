from crewai import Agent, Task, Crew, Process
import os
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
import gradio as gr


ollama_openhermes = Ollama(model='openhermes')
duckduckgo_search = DuckDuckGoSearchRun()

def create_crewai_setup(subject):
    # Define Agents
    researcher = Agent(
        role="Researcher",
        goal=f"""To explore and gather comprehensive information on {subject},
        distilling complex ideas into teachable insights.""",
        backstory=f"""You a curious and analytical individual with a background in academia.
        You have the passion for learning and an ability to dive deep into any subject matter, making connections that others might miss.
        Your expertise lies not only in gathering information but also in identifying the core principles that make a topic understandable for beginners.
        You have access to a wide range of resources and are adept at navigating academic and practical knowledge spheres to extract relevant information about {subject}.""",
        verbose=True,
        allow_delegation=False,
        tools=[duckduckgo_search],
        llm = ollama_openhermes
    )


    writer = Agent(
        role="write",
        goal=f"To translate the Researcher's ideas into an engaging and informative text that makes the topic on {subject} accessible to beginners.",
        backstory=f"""You are a talented communicator with a flair for making complex subjects on {subject} easy to understand.
        With a background in educational writing, excel at crafting clear, concise, and compelling narratives.
        You have a knack for storytelling and are skilled at using metaphors and analogies to illuminate difficult concepts.
        You are sensitive to the needs of their audience, always aiming to inform and inspire without overwhelming or patronizing.
        You view each piece of writing as an opportunity to spark curiosity and foster understanding.""",
        verbose=True,
        allow_delegation=False,
        llm = ollama_openhermes
    )

    examiner = Agent(
        role="Give examination",
        goal=f"To evaluate the reader's comprehension of the text through well-crafted test questions and answers.",
        backstory=f"""As an Examiner, you are an astute observer with a keen understanding of educational assessment.
        With experience in crafting and administering tests across various levels of learning,
        you are adept at measuring understanding and critical thinking.
        You knows how to ask the right questions to test not just memorization,
        but also the grasp of concepts and the ability to apply knowledge. You are meticulous in your approach,
        ensuring that each question serves a clear purpose in evaluating the reader's comprehension of the material.""",
        verbose=True,
        allow_delegation=False,
        llm = ollama_openhermes
    )

    # Create the tasks
    task1 = Task(description=f"Develop ideas for teaching someone new to {subject}.",
                 agent=researcher
                 )
   
    task2 = Task(description=f"Use the Researcher's ideas to write a piece of text to explain the topic on {subject}.",
                 agent=writer
                 )
   
    task3 = Task(description=f"""Craft 2-3 test questions to evaluate understanding of the created text on {subject}, along with the correct answers. 
                 In other words: test whether a student has fully understood the text.""",
                 agent=examiner
                 )
   
    # Create and Run the Crew
    crew = Crew(
        agents=[researcher, writer, examiner],
        tasks=[task1, task2, task3],
        verbose=2,
        process=Process.sequential
    )

    crew_result = crew.kickoff()
    return crew_result


# Gradio interface
def run_crewai_app(subject):
    crew_result = create_crewai_setup(subject)
    return crew_result

iface = gr.Interface(
    fn=run_crewai_app,
    inputs="text",
    outputs="text",
    title="CrewAI Educational Module Creation"
)

iface.launch(share=True)