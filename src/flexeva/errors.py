"""Typed failures exposed by the FlexEva public API."""

from __future__ import annotations


class FlexEvaError(Exception):
    """Base class for expected FlexEva failures."""


class PipelineContractError(FlexEvaError):
    """A layer or transform violated the RAS/AL contract."""


class SourceAnalysisError(FlexEvaError):
    """The mutable source cannot be analyzed soundly."""

    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        self.path = path
        self.line = line
        self.column = column
        location = ""
        if path is not None:
            location = path
            if line is not None:
                location += f":{line}"
                if column is not None:
                    location += f":{column}"
            location += ": "
        super().__init__(location + message)


class GroundingRequiredError(SourceAnalysisError):
    """A control decision needs Ground but no registered handler matches it."""


class AnchorNotFoundError(FlexEvaError):
    """The requested anchor reference is not present in the configured store."""


class StateStoreError(FlexEvaError):
    """Base class for managed-state storage failures."""


class StateNotFoundError(StateStoreError):
    """A managed state reference is not present in the configured store."""


class StateConflictError(StateStoreError):
    """A content address resolved to unequal managed state."""


class StateAccessError(StateStoreError):
    """A scoped reader attempted to access state outside its grant."""


class StateTransactionClosedError(StateStoreError):
    """A managed-state transaction was used after publish or abort."""


class StateStoreContractError(StateStoreError):
    """A record or reference violated the managed-state storage contract."""


class ConfigChangedError(FlexEvaError):
    """Refresh attempted to cross a pipeline or configuration boundary."""
