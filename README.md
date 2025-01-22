# dpbooster

A Python package that automates data management lifecycle.

## Features

### Core Features
- **Multi-Vendor Support**  
- **Cloud Agnostic**  
- **Multi-Provider RAG Support**  
- **Text Chat**: Interactive chat interfaces for all text models  
- **Voice Chat**: Real-time voice interactions with OpenAI models  
- **Video Chat**: Video processing capabilities with Gemini models  
- **Code Generation**: Specialized interfaces for coding assistance  
- **Multi-Modal**: Support for text, image, and video inputs  


### RAG Model Support

#### Core Language Models
| Provider | Models |
|----------|---------|
| OpenAI | gpt-4-turbo, gpt-4, gpt-3.5-turbo |
| Anthropic | claude-3-opus, claude-3-sonnet, claude-3-haiku |
| Gemini | gemini-pro, gemini-pro-vision, gemini-2.0-flash-exp |
| Groq | llama-3.2-70b-chat, mixtral-8x7b-chat |

#### Specialized Models
| Provider | Type | Models |
|----------|------|---------|
| DeepSeek | Multi-purpose | deepseek-chat, deepseek-coder, deepseek-vision |
| CrewAI | Agent Teams | Support Team, Article Team |


## Installation

### Basic Installation
```bash

pip install -e ".[all]" 
pip install -e ".[openai]" 
```

### API Key Configuration
```bash
# Core Providers
export OPENAI_API_KEY=<your token>
export GEMINI_API_KEY=<your token>
export ANTHROPIC_API_KEY=<your token>
export GROQ_API_KEY=<your token>

```

### Quick Start for RAG provider
```python
import gradio as gr
import components

# Create a simple RAG chat interface
gr.load(
    name='openai:gpt-4-turbo',  # or 'gemini:gemini-1.5-flash', 'groq:llama-3.2-70b-chat'
    src=components.registry,
    title='AI Chat',
    description='Chat with an AI model'
).launch()

# Create a chat interface with Transformers models
gr.load(
    name='transformers:phi-4',  # or 'transformers:tulu-3', 'transformers:olmo-2-13b'
    src=components.registry,
    title='Local AI Chat',
    description='Chat with locally running models'
).launch()

# Create a coding assistant with OpenAI
gr.load(
    name='openai:gpt-4-turbo',
    src=components.registry,
    coder=True,
    title='OpenAI Code Assistant',
    description='OpenAI Code Generator'
).launch()


```

### Advanced Features

#### Voice Chat
```python
gr.load(
    name='openai:gpt-4-turbo',
    src=components.registry,
    enable_voice=True,
    title='AI Voice Assistant'
).launch()
```

#### Camera Mode
```python
# Create a vision-enabled interface with camera support
gr.load(
    name='gemini:gemini-2.0-flash-exp',
    src=components.registry,
    camera=True,
).launch()
```

#### Multi-Provider Interface
```python
import gradio as gr
import components

with gr.Blocks() as demo:
    with gr.Tab("Text"):
        gr.load('openai:gpt-4-turbo', src=components.registry)
    with gr.Tab("Vision"):
        gr.load('gemini:gemini-pro-vision', src=components.registry)
    with gr.Tab("Code"):
        gr.load('deepseek:deepseek-coder', src=components.registry)

demo.launch()
```

#### CrewAI Teams
```python
# Article Creation Team
gr.load(
    name='crewai:gpt-4-turbo',
    src=components.registry,
    crew_type='article',
    title='AI Writing Team'
).launch()
```

#### Browser Automation

```bash
playwright install
```

use python 3.11+ for browser use

```python
import gradio as gr
import components

# Create a browser automation interface
gr.load(
    name='browser:gpt-4-turbo',
    src=components.registry,
    title='AI Browser Assistant',
    description='Let AI help with web tasks'
).launch()
```

Example tasks:
- Flight searches on Google Flights
- Weather lookups
- Product price comparisons
- News searches

#### Swarms Integration
```python
import gradio as gr
import components

# Create a chat interface with Swarms
gr.load(
    name='swarms:gpt-4-turbo',  # or other OpenAI models
    src=components.registry,
    agent_name="Stock-Analysis-Agent",  # customize agent name
    title='Swarms Chat',
    description='Chat with an AI agent powered by Swarms'
).launch()
```

#### Langchain Agents
```python
import gradio as gr
import components

# Create a Langchain agent interface
gr.load(
    name='langchain:gpt-4-turbo',  # or other supported models
    src=components.registry,
    title='Langchain Agent',
    description='AI agent powered by Langchain'
).launch()
```

## Requirements

### Core Requirements
- Python 3.10+
- gradio >= 5.9.1

### Optional Features
- Voice Chat: gradio-webrtc, numba==0.60.0, pydub, librosa
- Video Chat: opencv-python, Pillow
- Agent Teams: crewai>=0.1.0, langchain>=0.1.0

## Troubleshooting

### Authentication Issues
If you encounter 401 errors, verify your API keys:
```python
import os

# Set API keys manually if needed
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["GEMINI_API_KEY"] = "your-api-key"
```

### Provider Installation
If you see "no providers installed" errors:
```bash
# Install specific provider
pip install 'ai-gradio[provider_name]'

# Or install all providers
pip install 'ai-gradio[all]'
```


## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.






