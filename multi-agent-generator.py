import streamlit as st
from crewai import Agent as CrewAgent, Task as CrewTask, Crew
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import json
import time
from ibm_watsonx_ai.foundation_models import ModelInference

# For LangGraph
from langgraph.graph import StateGraph, END
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.agents import AgentFinish, AgentAction
from langchain_core.tools import BaseTool

# Load environment variables
load_dotenv()

class AgentGenerator:
    def __init__(self):
        # Initialize WatsonX AI model
        self.model_id = "meta-llama/llama-3-3-70b-instruct"
        self.parameters = {
            "decoding_method": "greedy",
            "max_new_tokens": 1000,  # Increased for complex JSON responses
            "min_new_tokens": 0,
            "repetition_penalty": 1
        }
        
        # Get project_id from environment
        self.project_id = os.getenv("PROJECT_ID")
        
        # Initialize model on first use instead of constructor
        self.model = None
    
    def _initialize_model(self):
        if self.model is None:
            credentials = {
                "url": "https://eu-de.ml.cloud.ibm.com",
                "apikey": os.getenv("WATSON_API_KEY")
            }
            
            if not credentials["apikey"]:
                st.warning("Watson API Key not found in environment. Please enter it below.")
                credentials["apikey"] = st.text_input("Enter Watson API Key:", type="password")
                if not credentials["apikey"]:
                    st.stop()
            
            self.model = ModelInference(
                model_id=self.model_id,
                params=self.parameters,
                credentials=credentials,
                project_id=self.project_id
            )
        
    def analyze_prompt(self, user_prompt: str, framework: str) -> Dict[str, Any]:
        self._initialize_model()
        
        system_prompt = self._get_system_prompt_for_framework(framework)
        
        try:
            # Format prompt for Llama-3
            formatted_prompt = f"""<|begin_of_text|>
<|system|>
{system_prompt}
<|user|>
{user_prompt}
<|assistant|>
"""
            
            # Generate response using WatsonX
            response = self.model.generate_text(prompt=formatted_prompt, guardrails=True)
            
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                st.warning("Could not extract valid JSON from model response. Using default configuration.")
                return self._get_default_config(framework)
                
        except Exception as e:
            st.error(f"Error in analyzing prompt: {e}")
            return self._get_default_config(framework)

    def _get_system_prompt_for_framework(self, framework: str) -> str:
        if framework == "crewai":
            return """
            You are an expert at creating AI research assistants using CrewAI. Based on the user's request,
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
        elif framework == "langgraph":
            return """
            You are an expert at creating AI agents using LangChain's LangGraph framework. Based on the user's request,
            suggest appropriate agents, their roles, tools, and nodes for the graph. Format your response as JSON with this structure:
            {
                "agents": [
                    {
                        "name": "agent name",
                        "role": "specific role description",
                        "goal": "clear goal",
                        "tools": ["tool1", "tool2"],
                        "llm": "model name (e.g., gpt-4)"
                    }
                ],
                "nodes": [
                    {
                        "name": "node name",
                        "description": "detailed description",
                        "agent": "agent name"
                    }
                ],
                "edges": [
                    {
                        "source": "source node name",
                        "target": "target node name",
                        "condition": "condition description (optional)"
                    }
                ]
            }
            """
        elif framework == "react":
            return """
            You are an expert at creating AI agents using the ReAct (Reasoning + Acting) framework. Based on the user's request,
            suggest appropriate agents, their roles, tools, and specific reasoning steps. Format your response as JSON with this structure:
            {
                "agents": [
                    {
                        "name": "agent name",
                        "role": "specific role description",
                        "goal": "clear goal",
                        "tools": ["tool1", "tool2"],
                        "llm": "model name (e.g., gpt-4)"
                    }
                ],
                "tools": [
                    {
                        "name": "tool name",
                        "description": "detailed description of what the tool does",
                        "parameters": {
                            "param1": "parameter description",
                            "param2": "parameter description"
                        }
                    }
                ],
                "examples": [
                    {
                        "query": "example user query",
                        "thought": "example thought process",
                        "action": "example action to take",
                        "observation": "example observation",
                        "final_answer": "example final answer"
                    }
                ]
            }
            """
        else:
            return """
            You are an expert at creating AI research assistants. Based on the user's request,
            suggest appropriate agents, their roles, tools, and tasks.
            """

    def _get_default_config(self, framework: str) -> Dict[str, Any]:
        if framework == "crewai":
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
        elif framework == "langgraph":
            return {
                "agents": [{
                    "name": "default_assistant",
                    "role": "General Assistant",
                    "goal": "Help with basic tasks",
                    "tools": ["basic_tool"],
                    "llm": "gpt-4"
                }],
                "nodes": [{
                    "name": "process_input",
                    "description": "Process user input",
                    "agent": "default_assistant"
                }],
                "edges": [{
                    "source": "process_input",
                    "target": "END",
                    "condition": "task completed"
                }]
            }
        elif framework == "react":
            return {
                "agents": [{
                    "name": "default_assistant",
                    "role": "General Assistant",
                    "goal": "Help with basic tasks",
                    "tools": ["basic_tool"],
                    "llm": "gpt-4"
                }],
                "tools": [{
                    "name": "basic_tool",
                    "description": "A basic utility tool",
                    "parameters": {
                        "input": "User input to process"
                    }
                }],
                "examples": [{
                    "query": "Help me find information",
                    "thought": "I need to search for relevant information",
                    "action": "Use search tool",
                    "observation": "Found relevant results",
                    "final_answer": "Here is the information you requested"
                }]
            }
        else:
            return {}


def create_code_block(config: Dict[str, Any], framework: str) -> str:
    if framework == "crewai":
        return create_crewai_code(config)
    elif framework == "langgraph":
        return create_langgraph_code(config)
    elif framework == "react":
        return create_react_code(config)
    else:
        return "# Invalid framework specified"


def create_crewai_code(config: Dict[str, Any]) -> str:
    code = "from crewai import Agent, Task, Crew\n\n"
    
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
    code += ")\n\n"
    code += "# Run the crew\n"
    code += "result = crew.kickoff()"
    
    return code


def create_langgraph_code(config: Dict[str, Any]) -> str:
    code = """from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from typing import Dict, List, Tuple, Any, TypedDict, Annotated
import operator

# Define state
class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: str

"""
    
    # Generate tool definitions if needed
    if any(agent["tools"] for agent in config["agents"]):
        code += "# Define tools\n"
        tools = set()
        for agent in config["agents"]:
            tools.update(agent["tools"])
        
        for tool in tools:
            code += f"""class {tool.capitalize()}Tool(BaseTool):
    name = "{tool}"
    description = "Tool for {tool} operations"
    
    def _run(self, query: str) -> str:
        # Implement actual functionality here
        return f"Result from {tool} tool: {{query}}"
    
    async def _arun(self, query: str) -> str:
        # Implement actual functionality here
        return f"Result from {tool} tool: {{query}}"

"""
        
        code += "tools = [\n"
        for tool in tools:
            code += f"    {tool.capitalize()}Tool(),\n"
        code += "]\n\n"
    
    # Generate Agent configurations
    for agent in config["agents"]:
        code += f"# Agent: {agent['name']}\n"
        code += f"def {agent['name']}_agent(state: AgentState) -> AgentState:\n"
        code += f"    \"\"\"Agent that handles {agent['role']}.\"\"\"\n"
        code += f"    # Create LLM\n"
        code += f"    llm = ChatOpenAI(model=\"{agent['llm']}\")\n"
        code += f"    # Get the most recent message\n"
        code += f"    messages = state['messages']\n"
        code += f"    response = llm.invoke(messages)\n"
        code += f"    # Add the response to the messages\n"
        code += f"    return {{\n"
        code += f"        \"messages\": messages + [response],\n"
        code += f"        \"next\": state.get(\"next\", \"\")\n"
        code += f"    }}\n\n"
    
    # Define routing logic function
    code += """# Define routing logic
def router(state: AgentState) -> str:
    \"\"\"Route to the next node.\"\"\"
    return state.get("next", "END")

"""
    
    # Generate graph configuration
    code += "# Define the graph\n"
    code += "workflow = StateGraph(AgentState)\n\n"
    
    # Add nodes
    code += "# Add nodes to the graph\n"
    for node in config["nodes"]:
        code += f"workflow.add_node(\"{node['name']}\", {node['agent']}_agent)\n"
    
    code += "\n# Add conditional edges\n"
    # Add edges
    for edge in config["edges"]:
        if edge["target"] == "END":
            code += f"workflow.add_edge(\"{edge['source']}\", END)\n"
        else:
            code += f"workflow.add_edge(\"{edge['source']}\", \"{edge['target']}\")\n"
    
    # Set entry point
    if config["nodes"]:
        code += f"\n# Set entry point\nworkflow.set_entry_point(\"{config['nodes'][0]['name']}\")\n"
    
    # Compile and run
    code += """
# Compile the graph
app = workflow.compile()

# Run the graph
def run_agent(query: str) -> List[BaseMessage]:
    \"\"\"Run the agent on a query.\"\"\"
    result = app.invoke({
        "messages": [HumanMessage(content=query)],
        "next": ""
    })
    return result["messages"]

# Example usage
if __name__ == "__main__":
    result = run_agent("Your query here")
    for message in result:
        print(f"{message.type}: {message.content}")
"""
    
    return code


def create_react_code(config: Dict[str, Any]) -> str:
    code = """from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from typing import Dict, List, Any

"""
    
    # Define tools
    code += "# Define tools\n"
    for tool in config.get("tools", []):
        code += f"""class {tool["name"].capitalize()}Tool(BaseTool):
    name = "{tool["name"]}"
    description = "{tool["description"]}"
    
    def _run(self, {", ".join(tool["parameters"].keys())}) -> str:
        # Implement actual functionality here
        return f"Result from {tool["name"]} tool"
    
    async def _arun(self, {", ".join(tool["parameters"].keys())}) -> str:
        # Implement actual functionality here
        return f"Result from {tool["name"]} tool"

"""
    
    # Collect tools
    code += "# Create tool instances\n"
    code += "tools = [\n"
    for tool in config.get("tools", []):
        code += f"    {tool['name'].capitalize()}Tool(),\n"
    code += "]\n\n"
    
    # Define example-based prompt
    code += "# Define example-based ReAct prompt\n"
    examples = config.get("examples", [])
    if examples:
        code += "examples = [\n"
        for example in examples:
            code += f"""    {{
        "query": "{example["query"]}",
        "thought": "{example["thought"]}",
        "action": "{example["action"]}",
        "observation": "{example["observation"]}",
        "final_answer": "{example["final_answer"]}"
    }},
"""
        code += "]\n\n"
    
    # Default agent
    if config.get("agents"):
        agent = config["agents"][0]  # Use the first agent
        code += f"""# Create ReAct agent
llm = ChatOpenAI(model="{agent["llm"]}")

# Create the agent using the ReAct framework
react_prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"You are {agent["role"]}. Your goal is to {agent["goal"]}.
    
Use the following tools to assist you:
{{tool_descriptions}}

Use the following format:
Question: The user question you need to answer
Thought: Consider what to do to best answer the question
Action: The action to take, should be one of {{tool_names}}
Action Input: The input to the action
Observation: The result of the action
... (Thought/Action/Action Input/Observation can repeat)
Thought: I now know the final answer
Final Answer: The final answer to the question\"\"\"),
    ("human", "{{input}}")
])

agent = create_react_agent(llm, tools, react_prompt)

# Create an agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent
def run_agent(query: str) -> str:
    \"\"\"Run the agent on a query.\"\"\"
    response = agent_executor.invoke({{"input": query}})
    return response.get("output", "No response generated")

# Example usage
if __name__ == "__main__":
    result = run_agent("Your query here")
    print(result)
"""
    
    return code


def main():
    st.set_page_config(page_title="Agent Framework Generator", page_icon="🚀", layout="wide")
    
    st.title("Multi-Framework Agent Generator")
    st.write("Generate agent code for different frameworks based on your requirements!")

    # Display IBM WatsonX AI information
    # st.sidebar.title("IBM Watsonx")
    st.sidebar.info("Powered by IBM Watsonx")
    
    # Check for API key
    if not os.getenv("WATSON_API_KEY"):
        api_key = st.sidebar.text_input("Watson API Key:", type="password")
        if api_key:
            os.environ["WATSON_API_KEY"] = api_key

    # Framework selection
    st.sidebar.title("🔄 Framework Selection")
    framework = st.sidebar.radio(
        "Choose a framework:",
        ["crewai", "langgraph", "react"],
        format_func=lambda x: {
            "crewai": "CrewAI",
            "langgraph": "LangGraph",
            "react": "ReAct Framework"
        }[x]
    )
    
    # Framework description
    framework_descriptions = {
        "crewai": """
        **CrewAI** is a framework for orchestrating role-playing autonomous AI agents. 
        It allows you to create a crew of agents that work together to accomplish tasks, 
        with each agent having a specific role, goal, and backstory.
        """,
        "langgraph": """
        **LangGraph** is LangChain's framework for building stateful, multi-actor applications with LLMs.
        It provides a way to create directed graphs where nodes are LLM calls, tools, or other operations, 
        and edges represent the flow of information between them.
        """,
        "react": """
        **ReAct** (Reasoning + Acting) is a framework that combines reasoning and action in LLM agents.
        It prompts the model to generate both reasoning traces and task-specific actions in an interleaved manner, 
        creating a synergy between the two that leads to improved performance.
        """
    }
    
    st.sidebar.markdown(framework_descriptions[framework])
    
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
        
        if st.button(f"🚀 Generate {framework.upper()} Code"):
            if not os.getenv("WATSON_API_KEY"):
                st.error("Please set your Watson API Key in the sidebar")
            elif not os.getenv("PROJECT_ID"):
                st.error("PROJECT_ID not found in .env file")
            else:
                with st.spinner(f"Generating your {framework} code..."):
                    generator = AgentGenerator()
                    config = generator.analyze_prompt(user_prompt, framework)
                    
                    # Store the configuration in session state
                    st.session_state.config = config
                    st.session_state.code = create_code_block(config, framework)
                    st.session_state.framework = framework
                    
                    time.sleep(0.5)  # Small delay for better UX
                    st.success(f"✨ {framework.upper()} code generated successfully!")

    with col2:
        st.subheader("💡 Framework Tips")
        if framework == "crewai":
            st.info("""
            **CrewAI Tips:**
            - Define clear roles for each agent
            - Set specific goals for better performance
            - Consider how agents should collaborate
            - Specify task delegation permissions
            """)
        elif framework == "langgraph":
            st.info("""
            **LangGraph Tips:**
            - Design your graph flow carefully
            - Define clear node responsibilities
            - Consider conditional routing between nodes
            - Think about how state is passed between nodes
            """)
        else:  # react
            st.info("""
            **ReAct Tips:**
            - Focus on the reasoning steps
            - Define tools with clear descriptions
            - Provide examples of thought processes
            - Consider the observation/action cycle
            """)

    # Display results
    if 'config' in st.session_state:
        st.subheader("🔍 Generated Configuration")
        
        # Tabs for different views
        tab1, tab2 = st.tabs(["📊 Visual Overview", "💻 Code"])
        
        with tab1:
            current_framework = st.session_state.framework
            
            if current_framework == "crewai":
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
            
            elif current_framework == "langgraph":
                # Display Agents
                st.subheader("Agents")
                for agent in st.session_state.config["agents"]:
                    with st.expander(f"🤖 {agent['role']}", expanded=True):
                        st.write(f"**Goal:** {agent['goal']}")
                        st.write(f"**Tools:** {', '.join(agent['tools'])}")
                        st.write(f"**LLM:** {agent['llm']}")
                
                # Display Nodes
                st.subheader("Graph Nodes")
                for node in st.session_state.config["nodes"]:
                    with st.expander(f"📍 {node['name']}", expanded=True):
                        st.write(f"**Description:** {node['description']}")
                        st.write(f"**Agent:** {node['agent']}")
                
                # Display Edges
                st.subheader("Graph Edges")
                for edge in st.session_state.config["edges"]:
                    with st.expander(f"🔗 {edge['source']} → {edge['target']}", expanded=True):
                        if "condition" in edge:
                            st.write(f"**Condition:** {edge['condition']}")
            
            elif current_framework == "react":
                # Display Agents
                st.subheader("Agents")
                for agent in st.session_state.config["agents"]:
                    with st.expander(f"🤖 {agent['role']}", expanded=True):
                        st.write(f"**Goal:** {agent['goal']}")
                        st.write(f"**Tools:** {', '.join(agent['tools'])}")
                        st.write(f"**LLM:** {agent['llm']}")
                
                # Display Tools
                st.subheader("Tools")
                for tool in st.session_state.config.get("tools", []):
                    with st.expander(f"🔧 {tool['name']}", expanded=True):
                        st.write(f"**Description:** {tool['description']}")
                        st.write("**Parameters:**")
                        for param, desc in tool["parameters"].items():
                            st.write(f"- **{param}**: {desc}")
                
                # Display Examples
                if "examples" in st.session_state.config:
                    st.subheader("Examples")
                    for i, example in enumerate(st.session_state.config["examples"]):
                        with st.expander(f"📝 Example {i+1}: {example['query'][:30]}...", expanded=True):
                            st.write(f"**Query:** {example['query']}")
                            st.write(f"**Thought:** {example['thought']}")
                            st.write(f"**Action:** {example['action']}")
                            st.write(f"**Observation:** {example['observation']}")
                            st.write(f"**Final Answer:** {example['final_answer']}")
        
        with tab2:
            st.code(st.session_state.code, language="python")
            if st.button("📋 Copy Code"):
                st.toast("Code copied to clipboard! 📋")

if __name__ == "__main__":
    main()