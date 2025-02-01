import gradio as gr

def custom_load(name: str, src: dict, **kwargs):
    # Only use custom loading if name contains provider prefix
    if ':' in name:
        provider, model = name.split(':')
        # Create provider-specific model key
        model_key = f"{provider}:{model}"
        
        if model_key not in src:
            available_models = [k for k in src.keys()]
            raise ValueError(f"Model {model_key} not found. Available models: {available_models}")
        return src[model_key](name=model, **kwargs)
    
    # Fall back to original gradio behavior if no provider prefix
    return original_load(name, src, **kwargs)

# Store original load function before overriding
original_load = gr.load
gr.load = custom_load

registry = {}

# OpenAI
try:
    from .openai_dpbooster import registry as openai_registry
    registry.update({f"openai:{k}": openai_registry for k in [
        "gpt-4o-2024-11-20",
        "gpt-4o",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-05-13",
        "chatgpt-4o-latest",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "o1-preview",
        "o1-preview-2024-09-12",
        "o1-mini",
        "o1-mini-2024-09-12",
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-turbo-preview",
        "gpt-4-0125-preview",
        "gpt-4-1106-preview",
        "gpt-4",
        "gpt-4-0613",
        "o1-2024-12-17",
        "gpt-4o-realtime-preview-2024-10-01",
        "gpt-4o-realtime-preview",
        "gpt-4o-realtime-preview-2024-12-17",
        "gpt-4o-mini-realtime-preview",
        "gpt-4o-mini-realtime-preview-2024-12-17",
    ]})
except ImportError as e:
    print(f"Failed to import OpenAI registry: {e}")

# Gemini
try:
    from .gemini_dpbooster import registry as gemini_registry
    registry.update({f"gemini:{k}": gemini_registry for k in [
        'gemini-1.5-flash',
        'gemini-1.5-flash-8b',
        'gemini-1.5-pro',
        'gemini-exp-1114',
        'gemini-exp-1121',
        'gemini-exp-1206',
        'gemini-2.0-flash-exp',
        'gemini-2.0-flash-thinking-exp-1219'
    ]})
except ImportError as e:
    print(f"Failed to import Gemini registry: {e}")

try:
    from .crewai_dpbooster import registry as crewai_registry
    # Add CrewAI models with their own prefix
    registry.update({f"crewai:{k}": crewai_registry for k in ['gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo']})
except ImportError:
    pass

try:
    from .anthropic_ import registry as anthropic_registry
    registry.update({f"anthropic:{k}": anthropic_registry for k in [
        'claude-3-5-sonnet-20241022',
        'claude-3-5-haiku-20241022',
        'claude-3-opus-20240229',
        'claude-3-sonnet-20240229',
        'claude-3-haiku-20240307',
    ]})
except ImportError:
    pass

     
        # Other Large Models
try:
    from .deepseek_dpbooster import registry as deepseek_registry
    registry.update({f"deepseek:{k}": deepseek_registry for k in [
        'deepseek-chat',
        'deepseek-coder',
        'deepseek-vision',
        'deepseek-reasoner'
    ]})
except ImportError:
    pass



try:
    from .groq_dpbooster import registry as groq_registry
    registry.update({f"groq:{k}": groq_registry for k in [
        "llama3-groq-8b-8192-tool-use-preview",
        "llama3-groq-70b-8192-tool-use-preview",
        "llama-3.2-1b-preview",
        "llama-3.2-3b-preview",
        "llama-3.2-11b-vision-preview",
        "llama-3.2-90b-vision-preview",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "gemma-7b-it",
        "llama-3.3-70b-versatile",
        "llama-3.3-70b-specdec"
    ]})
except ImportError:
    pass




try:
    from .transformers_dpbooster import registry as transformers_registry
    registry.update({f"transformers:{k}": transformers_registry for k in [
        "phi-4",
        "tulu-3",
        "olmo-2-13b",
        "smolvlm",
        "moondream",
        # Add other default transformers models here
    ]})
except ImportError:
    pass

try:
    from .jupyter_agent import registry as jupyter_registry
    registry.update({f"jupyter:{k}": jupyter_registry for k in [
        'meta-llama/Llama-3.2-3B-Instruct',
        'meta-llama/Llama-3.1-8B-Instruct', 
        'meta-llama/Llama-3.1-70B-Instruct'
    ]})
except ImportError:
    pass

try:
    from .langchain_dpbooster import registry as langchain_registry
    registry.update({f"langchain:{k}": langchain_registry for k in [
        'gpt-4-turbo',
        'gpt-4',
        'gpt-3.5-turbo',
        'gpt-3.5-turbo-0125'
    ]})
except ImportError as e:
    print(f"Failed to import LangChain registry: {e}")
    # Optionally add more detailed error handling here

try:
    from .mistral_dpbooster import registry as mistral_registry
    registry.update({f"mistral:{k}": mistral_registry for k in [
        "mistral-large-latest",
        "pixtral-large-latest",
        "ministral-3b-latest",
        "ministral-8b-latest",
        "mistral-small-latest",
        "codestral-latest",
        "mistral-embed",
        "mistral-moderation-latest",
        "pixtral-12b-2409",
        "open-mistral-nemo",
        "open-codestral-mamba",
    ]})
except ImportError:
    pass

try:
    from .nvidia_dpbooster import registry as nvidia_registry
    registry.update({f"nvidia:{k}": nvidia_registry for k in [
        "nvidia/llama3-chatqa-1.5-70b",
        "nvidia/cosmos-nemotron-34b",
        "nvidia/llama3-chatqa-1.5-8b",
        "nvidia-nemotron-4-340b-instruct",
        "meta/llama-3.1-70b-instruct",
        "meta/codellama-70b",
        "meta/llama2-70b",
        "meta/llama3-8b",
        "meta/llama3-70b",
        "mistralai/codestral-22b-instruct-v0.1",
        "mistralai/mathstral-7b-v0.1",
        "mistralai/mistral-large-2-instruct",
        "mistralai/mistral-7b-instruct",
        "mistralai/mistral-7b-instruct-v0.3",
        "mistralai/mixtral-8x7b-instruct",
        "mistralai/mixtral-8x22b-instruct",
        "mistralai/mistral-large",
        "google/gemma-2b",
        "google/gemma-7b",
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
        "google/codegemma-1.1-7b",
        "google/codegemma-7b",
        "google/recurrentgemma-2b",
        "google/shieldgemma-9b",
        "microsoft/phi-3-medium-128k-instruct",
        "microsoft/phi-3-medium-4k-instruct",
        "microsoft/phi-3-mini-128k-instruct",
        "microsoft/phi-3-mini-4k-instruct",
        "microsoft/phi-3-small-128k-instruct",
        "microsoft/phi-3-small-8k-instruct",
        "qwen/qwen2-7b-instruct",
        "databricks/dbrx-instruct",
        "deepseek-ai/deepseek-coder-6.7b-instruct",
        "upstage/solar-10.7b-instruct",
        "snowflake/arctic",
        "qwen/qwen2.5-7b-instruct"
    ]})
except ImportError:
    pass



try:
    from .kokoro_dpbooster import registry as kokoro_registry
    registry.update({f"kokoro:{k}": kokoro_registry for k in [
        "kokoro-v0_19"
    ]})
except ImportError:
    pass

if not registry:
    raise ImportError(
        "No providers installed. Install with either:\n"
        "pip install 'dpbooster[openai]' for OpenAI support\n"
        "pip install 'dpbooster[gemini]' for Gemini support\n"
        "pip install 'dpbooster[crewai]' for CrewAI support\n"
        "pip install 'dpbooster[anthropic]' for Anthropic support\n"
        "pip install 'dpbooster[lumaai]' for LumaAI support\n"
        "pip install 'dpbooster[xai]' for X.AI support\n"
        "pip install 'dpbooster[cohere]' for Cohere support\n"
        "pip install 'dpbooster[sambanova]' for SambaNova support\n"
        "pip install 'dpbooster[hyperbolic]' for Hyperbolic support\n"
        "pip install 'dpbooster[qwen]' for Qwen support\n"
        "pip install 'dpbooster[fireworks]' for Fireworks support\n"
        "pip install 'dpbooster[deepseek]' for DeepSeek support\n"
        "pip install 'dpbooster[smolagents]' for SmolaAgents support\n"
        "pip install 'dpbooster[jupyter]' for Jupyter support\n"
        "pip install 'dpbooster[langchain]' for LangChain support\n"
        "pip install 'dpbooster[all]' for all providers\n"
        "pip install 'dpbooster[swarms]' for Swarms support"
    )

__all__ = ["registry"]
