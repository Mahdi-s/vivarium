import torch
if not hasattr(torch.library, "register_fake"):
    print("Mocking torch.library.register_fake")
    def register_fake(name):
        def decorator(fn):
            return fn
        return decorator
    torch.library.register_fake = register_fake

import transformers
from transformers import AutoConfig, AutoModelForCausalLM

model_id = "allenai/Olmo-3-1025-7B"
try:
    print(f"Transformers version: {transformers.__version__}")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    # Sanitize config for rope_scaling
    rs = getattr(config, "rope_scaling", None)
    if isinstance(rs, dict):
        for k in ["beta_fast", "beta_slow"]:
            if k in rs:
                rs[k] = float(rs[k])
    
    model = AutoModelForCausalLM.from_pretrained(model_id, config=config, trust_remote_code=True, device_map="cpu")
    print(f"Successfully loaded model: {type(model)}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
