import transformers
from transformers import AutoConfig, AutoModelForCausalLM

model_id = "allenai/Olmo-3-1025-7B"
try:
    print(f"Transformers version: {transformers.__version__}")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print(f"Config class: {type(config)}")
    print(f"Architectures: {getattr(config, 'architectures', 'N/A')}")
    
    # Try importing the class directly
    try:
        from transformers.models.olmo3.modeling_olmo3 import Olmo3ForCausalLM
        print("Successfully imported Olmo3ForCausalLM directly")
    except ImportError as e:
        print(f"Failed to import Olmo3ForCausalLM directly: {e}")
        
    model = AutoModelForCausalLM.from_pretrained(model_id, config=config, trust_remote_code=True, device_map="cpu")
    print(f"Successfully loaded model: {type(model)}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
