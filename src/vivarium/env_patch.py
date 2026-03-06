import sys

def apply_patches() -> None:
    """
    Apply optional environment patches.

    IMPORTANT: Do not import heavy optional deps (like torch) at import time.
    Only patch them if they are already imported by the caller.
    """
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if hasattr(torch, "library") and not hasattr(torch.library, "register_fake"):
                def _register_fake(name):  # type: ignore[no-untyped-def]
                    def decorator(fn):  # type: ignore[no-untyped-def]
                        return fn
                    return decorator
                torch.library.register_fake = _register_fake  # type: ignore[attr-defined]
        except Exception:
            pass

    # Patch torch._dynamo.utils if present (do not import it here).
    du = sys.modules.get("torch._dynamo.utils")
    if du is not None:
        try:
            if not hasattr(du, "warn_once"):
                def _warn_once(msg):  # type: ignore[no-untyped-def]
                    return None
                du.warn_once = _warn_once  # type: ignore[attr-defined]
        except Exception:
            pass


apply_patches()

# Force mock certain modules if they are known to be broken in this env
# This is a bit aggressive but might be needed
if "torchvision" in sys.modules:
    # Too late if already imported, but let's try to prevent future issues
    pass
