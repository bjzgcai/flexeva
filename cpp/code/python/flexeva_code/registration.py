"""Backend-neutral registration of a constrained Code execution window."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CodeRegistration:
    """Rules observe actual calls; the evaluator supplies effects and lifecycle.

    The Code layer does not construct evaluator sessions or interpret emitted
    native payloads. An adapter may extend this registration with a session
    factory and backend-specific rules.
    """
    rules: object
    identity: str = "restricted-python-source-v1"
    modeled_function: str | None = None
    complete_invocation: bool = False
    interface_observer_factory: object = None
    execution_context_factory: object = None
    context_identity: object = None
