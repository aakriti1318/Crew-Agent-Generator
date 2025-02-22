import streamlit as st
from crewai import Agent, Task, Crew
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import time

# Load environment variables
load_dotenv()

class AgentGenerator:
    def __init__(self):
        self.client = OpenAI(api_key="openai_api_key")
        
    def analyze_prompt(self, user_prompt: str) -> Dict[str, Any]:
        system_prompt = """
        You are an expert at creating AI research assistants. Based on the user's request,
        suggest appropriate agents, their roles, tools, and tasks. Format your response as JSON with this structure:
        {
            "agents": [
                {
                    "name": "agent name",
                    "role": "specific role description",
                    "goal": "clear goal",
                    "backstory": "relevant backstory",
                    "tools": ["tool1", "tool2"],
                    "verbose": true,
                    "allow_delegation": true/false
                }
            ],
            "tasks": [
                {
                    "name": "task name",
                    "description": "detailed description",
                    "tools": ["required tools"],
                    "agent": "agent name",
                    "expected_output": "specific expected output"
                }
            ]
        }
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error in analyzing prompt: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "agents": [{
                "name": "default_assistant",
                "role": "General Assistant",
                "goal": "Help with basic tasks",
                "backstory": "Versatile assistant with general knowledge",
                "tools": ["basic_tool"],
                "verbose": True,
                "allow_delegation": False
            }],
            "tasks": [{
                "name": "basic_task",
                "description": "Handle basic requests",
                "tools": ["basic_tool"],
                "agent": "default_assistant",
                "expected_output": "Task completion"
            }]
        }

def create_code_block(config: Dict[str, Any]) -> str:
    code = ""
    
    # Generate Agent configurations
    for agent in config["agents"]:
        code += f"# Agent: {agent['name']}\n"
        code += f"agent_{agent['name']} = Agent(\n"
        code += f"    role='{agent['role']}',\n"
        code += f"    goal='{agent['goal']}',\n"
        code += f"    backstory='{agent['backstory']}',\n"
        code += f"    verbose={agent['verbose']},\n"
        code += f"    allow_delegation={agent['allow_delegation']},\n"
        code += f"    tools={agent['tools']}\n"
        code += ")\n\n"

    # Generate Task configurations
    for task in config["tasks"]:
        code += f"# Task: {task['name']}\n"
        code += f"task_{task['name']} = Task(\n"
        code += f"    description='{task['description']}',\n"
        code += f"    agent=agent_{task['agent']},\n"
        code += f"    expected_output='{task['expected_output']}'\n"
        code += ")\n\n"

    # Generate Crew configuration
    code += "# Crew Configuration\n"
    code += "crew = Crew(\n"
    code += "    agents=[" + ", ".join(f"agent_{a['name']}" for a in config["agents"]) + "],\n"
    code += "    tasks=[" + ", ".join(f"task_{t['name']}" for t in config["tasks"]) + "]\n"
    code += ")"
    
    return code

def main():
    st.set_page_config(page_title="CrewAI Generator", page_icon="🤖", layout="wide")
    
    st.title("🤖 CrewAI Agent Generator")
    st.write("Generate custom AI agent crews based on your requirements!")

    # Sidebar for examples
    st.sidebar.title("📚 Example Prompts")
    example_prompts = {
        "Research Assistant": "I need a research assistant that summarizes papers and answers questions",
        "Content Creation": "I need a team to create viral social media content and manage our brand presence",
        "Data Analysis": "I need a team to analyze customer data and create visualizations",
        "Technical Writing": "I need a team to create technical documentation and API guides"
    }
    
    selected_example = st.sidebar.selectbox("Choose an example:", list(example_prompts.keys()))
    
    # Main input area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Define Your Requirements")
        user_prompt = st.text_area(
            "Describe what you need:",
            value=example_prompts[selected_example],
            height=100
        )
        
        if st.button("🚀 Generate Crew"):
            with st.spinner("Generating your custom AI crew..."):
                generator = AgentGenerator()
                config = generator.analyze_prompt(user_prompt)
                
                # Store the configuration in session state
                st.session_state.config = config
                st.session_state.code = create_code_block(config)
                
                time.sleep(0.5)  # Small delay for better UX
                st.success("✨ Crew generated successfully!")

    with col2:
        st.subheader("💡 Tips")
        st.info("""
        - Be specific about the tasks you need
        - Mention any special tools required
        - Specify if you need multiple agents
        - Include any particular goals or constraints
        """)

    # Display results
    if 'config' in st.session_state:
        st.subheader("🔍 Generated Configuration")
        
        # Tabs for different views
        tab1, tab2 = st.tabs(["📊 Visual Overview", "💻 Code"])
        
        with tab1:
            # Display Agents
            st.subheader("Agents")
            for agent in st.session_state.config["agents"]:
                with st.expander(f"🤖 {agent['role']}", expanded=True):
                    st.write(f"**Goal:** {agent['goal']}")
                    st.write(f"**Backstory:** {agent['backstory']}")
                    st.write(f"**Tools:** {', '.join(agent['tools'])}")
            
            # Display Tasks
            st.subheader("Tasks")
            for task in st.session_state.config["tasks"]:
                with st.expander(f"📋 {task['name']}", expanded=True):
                    st.write(f"**Description:** {task['description']}")
                    st.write(f"**Expected Output:** {task['expected_output']}")
                    st.write(f"**Assigned to:** {task['agent']}")
        
        with tab2:
            st.code(st.session_state.code, language="python")
            if st.button("📋 Copy Code"):
                st.toast("Code copied to clipboard! 📋")

if __name__ == "__main__":
    main()