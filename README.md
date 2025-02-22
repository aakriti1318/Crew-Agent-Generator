# Crew Agent Generator

## Overview
This is a **CrewAI Generator** built using **Streamlit** that allows users to generate AI agent configurations dynamically. The tool leverages **OpenAI's GPT-4o-mini** to analyze user requirements and create a structured CrewAI setup, including agents, tasks, and necessary tools.

<video width="600" controls>
  <source src="https://raw.githubusercontent.com/aakriti1318/Crew-Agent-Generator/blob/main/crew_generator.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

[![Watch the demo](https://github.com/aakriti1318/Crew-Agent-Generator/blob/main/agent.png)](https://github.com/aakriti1318/Crew-Agent-Generator/issues/1#issue-2870789620)


## Features
- Accepts simple English input to define AI agents and tasks
- Automatically generates a structured AI workflow
- Provides ready-to-use Python code for CrewAI
- Built using **Streamlit** for an interactive UI

## Installation
To set up and run this application, follow these steps:

### 1. Clone the repository:
```sh
git clone https://github.com/aakriti1318/Crew-Agent-Generator.git
cd Crew-Agent-Generator
```

### 2. Install dependencies:
Install the required dependencies:
```sh
pip install -r requirements.txt
```

### 3. Set up OpenAI API Key
Create a `.env` file in the root directory and add your OpenAI API key:
```sh
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```
Alternatively, manually create the `.env` file and add:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the application
Start the Streamlit app:
```sh
streamlit run main.py
```

This will launch the application in your default web browser.

## Usage
1. Open the Streamlit app in your browser.
2. Enter a prompt describing your AI agent requirements.
3. Click on **Generate Crew** to create a structured AI workflow.
4. View the generated code and copy it for your project.

## Contributing
Feel free to fork the repository, submit pull requests, and suggest improvements!

