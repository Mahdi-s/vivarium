import sys

# Patch torch.library if it exists but lacks register_fake
try:
    import torch
    if hasattr(torch, "library") and not hasattr(torch.library, "register_fake"):
        def _register_fake(name):
            def decorator(fn): return fn
            return decorator
        torch.library.register_fake = _register_fake
except ImportError:
    pass

# Patch torch._dynamo.utils
try:
    import torch._dynamo.utils as _du
    if not hasattr(_du, "warn_once"):
        def _warn_once(msg): pass
        _du.warn_once = _warn_once
except (ImportError, AttributeError):
    # If the module itself is missing or part of the path is missing
    pass

# Force mock certain modules if they are known to be broken in this env
# This is a bit aggressive but might be needed
if "torchvision" in sys.modules:
    # Too late if already imported, but let's try to prevent future issues
    pass
