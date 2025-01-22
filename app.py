import components
import gradio as gr

# Print available models
print("Available models:", list(components.registry.keys()))



gr.load(
    name='openai:gpt-4-turbo',  # or another model from the available list
    src=components.registry,
    title='AI Chat',
    description='Draw KPIs with an AI model'
).launch()


