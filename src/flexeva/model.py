"""Backend-neutral immutable records used by the RAS/AL runtime."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeAlias, TypeVar


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    raise TypeError(f"value is not deterministically fingerprintable: {type(value).__name__}")


def stable_digest(value: Any) -> str:
    """Return a deterministic content digest without defining a storage codec."""

    encoded = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class WindowRef:
    """Stable identity shared by every managed state record in one evaluation window."""

    value: str

    @classmethod
    def create(cls, config_digest: str, pipeline_digest: str) -> WindowRef:
        """Derive a window without incorporating a mutable-region revision."""

        return cls(f"window-{stable_digest((config_digest, pipeline_digest))}")

    def __str__(self) -> str:
        return self.value


class StateKind(str, Enum):
    """Runtime schema tags for content-addressed managed state."""

    ROOT = "root"
    BACKEND = "backend"
    PAYLOAD = "payload"
    PARTITION_INPUT = "partition-input"
    PARTITION = "partition"
    LAYER = "layer"
    TRANSFORM = "transform"
    GROUND = "ground"
    CANDIDATE = "candidate"
    ANCHOR = "anchor"
    INDEX_PAGE = "index-page"


StateRecordT = TypeVar("StateRecordT")


@dataclass(frozen=True)
class StateRef(Generic[StateRecordT]):
    """Typed, window-scoped reference to one immutable managed state record."""

    kind: StateKind
    window: WindowRef
    digest: str

    @property
    def value(self) -> str:
        return f"{self.kind.value}:{self.window.value}:{self.digest}"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class BackendState:
    """Unpublished backend-native state delivered by capture or Apply.

    This is an input to admission, never a mutable handle into published state.
    Explicit references declare shared backing state without interpreting its format.
    """

    value: Any
    state_refs: tuple[StateRef[Any], ...] = ()


@dataclass(frozen=True)
class BackendStateRecord:
    """One immutable backing snapshot; its representation belongs to the backend."""

    kind: ClassVar[StateKind] = StateKind.BACKEND

    window: WindowRef
    layer_id: str
    value: Any
    state_refs: tuple[StateRef[Any], ...] = ()


@dataclass(frozen=True)
class PayloadBinding:
    """A backing snapshot and an opaque, backend-interpreted event locator.

    Inline BackendState is owned by the containing partition snapshot. Shared
    state may use a managed reference. Context is capture state, not an input key.
    """

    state: BackendState | StateRef[BackendStateRecord]
    locator: Any = None
    context_ref: StateRef[BackendStateRecord] | None = None

    @classmethod
    def inline(cls, value: Any, locator: Any = None) -> PayloadBinding:
        return cls(BackendState(value), locator)


class LineageKind(str, Enum):
    BOUNDARY = "boundary"
    DATA = "data"


@dataclass(frozen=True)
class SourceSpan:
    path: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int


@dataclass(frozen=True)
class FrozenPayload:
    """Backend-encoded immutable body. Core never decodes this packet."""

    schema: str
    data: bytes
    state_refs: tuple[StateRef[Any], ...] = ()

    def __post_init__(self):
        if not self.schema or type(self.data) is not bytes or not isinstance(self.state_refs, tuple):
            raise TypeError("FrozenPayload requires a schema, immutable bytes and explicit references")


@dataclass(frozen=True)
class PayloadBlockRecord:
    kind: ClassVar[StateKind] = StateKind.PAYLOAD
    window: WindowRef
    layer_id: str
    body: FrozenPayload


@dataclass(frozen=True)
class PayloadHandle:
    """A layer-bound packet or published packet reference; locator is backend-owned."""

    layer_id: str
    body: FrozenPayload | StateRef[PayloadBlockRecord]
    locator: bytes = b""
    context_ref: StateRef[BackendStateRecord] | None = None


@dataclass(frozen=True)
class SourceFile:
    path: str
    text: str
    digest: str

    @classmethod
    def from_text(cls, path: str, text: str) -> SourceFile:
        return cls(path=path, text=text, digest=stable_digest(text))


@dataclass(frozen=True)
class SourceSnapshot:
    files: tuple[SourceFile, ...]
    digest: str

    @classmethod
    def create(cls, files: tuple[SourceFile, ...]) -> SourceSnapshot:
        ordered = tuple(sorted(files, key=lambda item: item.path))
        return cls(
            files=ordered,
            digest=stable_digest(tuple((item.path, item.digest) for item in ordered)),
        )

    def by_path(self) -> dict[str, SourceFile]:
        return {item.path: item for item in self.files}


@dataclass(frozen=True)
class LaneContext:
    active_lanes: tuple[int | str, ...]


@dataclass(frozen=True)
class NeedGround:
    """Lane asks Core to bind a query to immutable construction inputs."""

    request_id: str
    point_id: str
    requested_lanes: tuple[int | str, ...]
    query: Any
    inputs: Any
    input_refs: tuple[StateRef[Any], ...] = ()
    provenance: SourceSpan | None = None


@dataclass(frozen=True)
class EventPayload:
    """Transient result of Payload: actual content and its semantic identity."""
    value: Any
    fingerprint: str

    @classmethod
    def create(cls, value: Any) -> EventPayload:
        return cls(value=value, fingerprint=stable_digest(value))


@dataclass(frozen=True)
class Event:
    """Logical metadata with a native-state binding, never a cached decoded body."""

    event_id: str
    kind: str
    binding: PayloadBinding
    scope_id: str
    match_key: str
    provenance: SourceSpan | None = None
    origin_boundary_id: str | None = None
    lane_context: LaneContext | None = None
    payload_fingerprint: str | None = None
    defs: tuple[str, ...] = ()
    uses: tuple[str, ...] = ()
    layer_id: str = ""

    @property
    def semantic_key(self) -> tuple:
        return (
            self.kind,
            (tuple(sorted(self.lane_context.active_lanes,
                          key=lambda lane: (type(lane).__name__, str(lane))))
             if self.lane_context is not None else None),
            self.payload_fingerprint,
            self.defs,
            self.uses,
            self.origin_boundary_id,
        )

    @property
    def fingerprint(self) -> str:
        return stable_digest(self.semantic_key)


@dataclass(frozen=True)
class EventCapture:
    """One immutable observation produced by the root RAS adapter."""

    revision: str
    events: tuple[Event, ...]
    evidence: Any


@dataclass(frozen=True)
class EventDependency:
    producer_event_id: str
    consumer_event_id: str
    kind: str = "data"
    symbols: tuple[str, ...] = ()

    @property
    def edge_id(self) -> str:
        return stable_digest(
            (self.producer_event_id, self.consumer_event_id, self.kind, self.symbols)
        )[:24]


@dataclass(frozen=True)
class EventPartition:
    """Optional evaluator assembly value, never a Core Store schema.

    The evaluator groups and checks descriptors, then supplies an opaque
    RASPartition. Core never receives this event membership.
    """

    partition_id: str
    event_ids: tuple[str, ...]
    state: Any = None
    events: tuple[Event, ...] = ()
    lane_signature: tuple[int | str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.event_ids, tuple):
            raise TypeError("partition event_ids must be a tuple")
        if self.events:
            event_ids = tuple(event.event_id for event in self.events)
            if self.event_ids and self.event_ids != event_ids:
                raise ValueError("partition event_ids disagree with embedded events")
            object.__setattr__(self, "event_ids", event_ids)
            signatures = {
                tuple(sorted(event.lane_context.active_lanes,
                             key=lambda lane: (type(lane).__name__, str(lane))))
                for event in self.events
                if event.lane_context is not None
            }
            if not self.lane_signature and len(signatures) == 1:
                object.__setattr__(self, "lane_signature", next(iter(signatures)))

    @property
    def layout_fingerprint(self) -> str:
        return stable_digest((self.event_ids, self.lane_signature))

    @property
    def state_fingerprint(self) -> str:
        return stable_digest(self.state)

    @property
    def fingerprint(self) -> str:
        return stable_digest((self.layout_fingerprint, self.state_fingerprint,
                              tuple(event.semantic_key for event in self.events)))


@dataclass(frozen=True)
class RootCapture:
    """Frozen root input, with no Core-visible event collection."""

    revision: str
    payload: PayloadHandle
    evidence: Any = None


@dataclass(frozen=True)
class DependencyEdge:
    producer_partition_id: str
    consumer_partition_id: str
    kind: str = "data"
    symbols: tuple[str, ...] = ()

    @property
    def edge_id(self):
        return stable_digest((self.producer_partition_id, self.consumer_partition_id, self.kind))[:24]


@dataclass(frozen=True)
class RASPartition:
    """Complete opaque partition, not an event container in Core."""

    partition_id: str
    binding: PayloadHandle
    lane_signature: tuple[int | str, ...]
    event_count: int
    event_kinds: tuple[str, ...]
    semantic_fingerprint: str
    layout_fingerprint: str
    state_fingerprint: str
    origins: tuple[str | None, ...] = ()

    @property
    def fingerprint(self):
        return stable_digest((self.semantic_fingerprint, self.layout_fingerprint, self.state_fingerprint))


@dataclass(frozen=True)
class BoundaryRecord:
    """Semantic inputs and current ownership, not a native execution schedule."""

    boundary_id: str
    upper_partition_id: str
    input_partition_ids: tuple[str, ...]
    payload: PayloadHandle
    semantic_fingerprint: str
    state_refs: tuple[StateRef[Any], ...] = ()

    @property
    def fingerprint(self) -> str:
        return stable_digest((self.input_partition_ids, self.semantic_fingerprint, self.state_refs))


@dataclass(frozen=True)
class BoundaryMaterialization:
    """One coarse generation receipt; finalized RAS ownership lives below."""

    boundary_id: str
    input_key: str
    input_ref: StateRef[PartitionInputRecord]
    event_count: int
    state: PayloadHandle | None = None


@dataclass(frozen=True)
class GeneratedReceipt:
    """Opaque coarse output. Native ordering is encoded by the evaluator."""

    boundary_id: str
    binding: PayloadHandle
    event_count: int
    state: PayloadHandle | None = None


@dataclass(frozen=True)
class ApplyResult:
    generated: tuple[GeneratedReceipt, ...]
    removed_receipt_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ALEdge:
    edge_id: str
    upper_partition_id: str
    lower_partition_id: str
    boundary_id: str | None
    kind: LineageKind


@dataclass(frozen=True)
class LayerSnapshot:
    layer_id: str
    partitions: tuple[RASPartition, ...]
    dependencies: tuple[DependencyEdge, ...]
    state: Any = None

    def partitions_by_id(self):
        return {part.partition_id: part for part in self.partitions}


@dataclass(frozen=True)
class TransformSnapshot:
    transform_id: str
    boundaries: tuple[BoundaryRecord, ...]
    lineage: tuple[ALEdge, ...]


@dataclass(frozen=True)
class LayerDelta:
    layer_id: str
    added_partition_ids: tuple[str, ...] = ()
    changed_partition_ids: tuple[str, ...] = ()
    removed_partition_ids: tuple[str, ...] = ()
    invalidated_partition_ids: tuple[str, ...] = ()
    updated_partition_ids: tuple[str, ...] = ()
    partition_order_changed: bool = False
    dependencies_changed: bool = False
    layout_changed_partition_ids: tuple[str, ...] = ()
    state_changed_partition_ids: tuple[str, ...] = ()
    layer_state_changed: bool = False

    @property
    def affected_partition_ids(self):
        return tuple(dict.fromkeys(self.added_partition_ids + self.changed_partition_ids
                                   + self.removed_partition_ids + self.invalidated_partition_ids))

    @property
    def has_semantic_changes(self):
        return bool(self.affected_partition_ids or self.dependencies_changed)

    @property
    def is_empty(self):
        return not (self.affected_partition_ids or self.updated_partition_ids
                    or self.partition_order_changed or self.dependencies_changed
                    or self.layout_changed_partition_ids or self.state_changed_partition_ids
                    or self.layer_state_changed)


@dataclass(frozen=True)
class RefreshDelta:
    layers: tuple[LayerDelta, ...]

    def for_layer(self, layer_id: str) -> LayerDelta:
        for layer in self.layers:
            if layer.layer_id == layer_id:
                return layer
        raise KeyError(layer_id)


@dataclass(frozen=True)
class RootRecord:
    """Closed root observation with exact input events and shared native context."""

    kind: ClassVar[StateKind] = StateKind.ROOT

    window: WindowRef
    revision: str
    evidence: StateRef[BackendStateRecord]
    input_ref: StateRef[PartitionInputRecord] | None = None


@dataclass(frozen=True)
class PartitionInputRecord:
    """Coarse input body binding; never an indexed event sequence."""

    kind: ClassVar[StateKind] = StateKind.PARTITION_INPUT
    window: WindowRef
    layer_id: str
    boundary_id: str | None
    binding: PayloadHandle
    event_count: int = 0


@dataclass(frozen=True)
class PartitionRecord:
    """One immutable partition snapshot, including its event descriptors."""

    kind: ClassVar[StateKind] = StateKind.PARTITION

    window: WindowRef
    layer_id: str
    partition: RASPartition


@dataclass(frozen=True)
class GroundRequest:
    """Store-bound query; independent of final event identity and placement."""

    request_id: str
    layer_id: str
    point_id: str
    requested_lanes: tuple[int | str, ...]
    query: Any
    input_ref: StateRef[BackendStateRecord]


@dataclass(frozen=True)
class GroundResult:
    """Evaluator output, optionally safe to reuse across refreshes in its window."""

    request_id: str
    payload: Any
    reusable: bool = False


@dataclass(frozen=True)
class LayerManifest:
    """Independent immutable directory roots, not a full partition/event list."""

    kind: ClassVar[StateKind] = StateKind.LAYER

    window: WindowRef
    layer_id: str
    indexes: LayerIndexes
    state: Any = None


@dataclass(frozen=True)
class TransformManifest:
    """Boundary/AL metadata with opaque generation receipts."""

    kind: ClassVar[StateKind] = StateKind.TRANSFORM

    window: WindowRef
    transform_id: str
    upper_layer_ref: StateRef[LayerManifest]
    lower_layer_ref: StateRef[LayerManifest]
    indexes: TransformIndexes


@dataclass(frozen=True)
class GroundRecord:
    """Auditable result of resolving one Lane request through a workload rule."""

    kind: ClassVar[StateKind] = StateKind.GROUND

    window: WindowRef
    registration_id: str
    cache_key: str
    request: GroundRequest
    result: GroundResult

    @property
    def layer_id(self) -> str:
        return self.request.layer_id

    @property
    def reusable(self) -> bool:
        return self.result.reusable


@dataclass(frozen=True)
class GroundResolution:
    """Reference granted to Lane when Core resumes after Ground."""

    request_id: str
    ground_ref: StateRef[GroundRecord]


@dataclass(frozen=True)
class CandidateManifest:
    """Store-resident refresh result awaiting promotion to an anchor."""

    kind: ClassVar[StateKind] = StateKind.CANDIDATE

    window: WindowRef
    parent: StateRef[AnchorManifest]
    config_digest: str
    pipeline_digest: str
    root_ref: StateRef[RootRecord]
    layer_refs: tuple[StateRef[LayerManifest], ...]
    transform_refs: tuple[StateRef[TransformManifest], ...]
    ground_refs: tuple[StateRef[GroundRecord], ...]
    delta: RefreshDelta


@dataclass(frozen=True)
class AnchorManifest:
    """Committed manifest whose payload consists entirely of managed state references."""

    kind: ClassVar[StateKind] = StateKind.ANCHOR

    window: WindowRef
    config_digest: str
    pipeline_digest: str
    root_ref: StateRef[RootRecord]
    layer_refs: tuple[StateRef[LayerManifest], ...]
    transform_refs: tuple[StateRef[TransformManifest], ...]
    ground_refs: tuple[StateRef[GroundRecord], ...] = ()
    parent: StateRef[AnchorManifest] | None = None


@dataclass(frozen=True)
class IndexChild:
    """Routing metadata; consulting it never opens a partition body."""

    maximum: Any
    ref: StateRef[IndexPageRecord]
    count: int
    width: int


@dataclass(frozen=True)
class IndexPageRecord:
    """An immutable B+ tree page, never an independently stored Event."""

    kind: ClassVar[StateKind] = StateKind.INDEX_PAGE
    window: WindowRef
    namespace: str
    entries: tuple[tuple[Any, Any], ...] = ()
    children: tuple[IndexChild, ...] = ()
    value_refs: tuple[StateRef[Any], ...] = ()


@dataclass(frozen=True)
class IndexRoot:
    namespace: str
    ref: StateRef[IndexPageRecord] | None = None
    count: int = 0


@dataclass(frozen=True)
class PartitionEntry:
    """Directory entry: identity and metadata, never native event bodies."""

    partition_id: str
    ref: StateRef[PartitionRecord]
    lane_signature: tuple[int | str, ...]
    origins: tuple[str | None, ...]
    semantic_fingerprint: str
    layout_fingerprint: str
    state_fingerprint: str
    event_count: int
    event_kinds: tuple[str, ...]


@dataclass(frozen=True)
class OrderSplice:
    """One edit in the *parent* sequence's coordinates."""

    start: int
    delete_count: int
    partition_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PartitionDelta:
    """Backend-owned complete new bodies and local ownership/order changes."""

    upserts: tuple[RASPartition, ...] = ()
    removed_ids: tuple[str, ...] = ()
    order_splices: tuple[OrderSplice, ...] = ()


@dataclass(frozen=True)
class DependencyDelta:
    upserts: tuple[DependencyEdge, ...] = ()
    removed_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class LayerIndexes:
    partitions: IndexRoot = field(default_factory=lambda: IndexRoot("partitions"))
    order: IndexRoot = field(default_factory=lambda: IndexRoot("order"))
    positions: IndexRoot = field(default_factory=lambda: IndexRoot("positions"))
    origins: IndexRoot = field(default_factory=lambda: IndexRoot("origins"))
    dependencies: IndexRoot = field(default_factory=lambda: IndexRoot("dependencies"))
    consumers: IndexRoot = field(default_factory=lambda: IndexRoot("consumers"))
    incoming: IndexRoot = field(default_factory=lambda: IndexRoot("incoming"))
    semantics: IndexRoot = field(default_factory=lambda: IndexRoot("semantics"))


@dataclass(frozen=True)
class TransformIndexes:
    boundaries: IndexRoot = field(default_factory=lambda: IndexRoot("boundaries"))
    receipts: IndexRoot = field(default_factory=lambda: IndexRoot("receipts"))
    lineage: IndexRoot = field(default_factory=lambda: IndexRoot("lineage"))
    producers: IndexRoot = field(default_factory=lambda: IndexRoot("producers"))
    boundary_inputs: IndexRoot = field(default_factory=lambda: IndexRoot("boundary-inputs"))


@dataclass(frozen=True)
class BoundaryDelta:
    upserts: tuple[BoundaryRecord, ...] = ()
    removed_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class LinkDelta:
    upserts: tuple[ALEdge, ...] = ()
    removed_ids: tuple[str, ...] = ()


ManagedStateRecord: TypeAlias = (
    BackendStateRecord
    | PayloadBlockRecord
    | RootRecord
    | PartitionInputRecord
    | PartitionRecord
    | LayerManifest
    | TransformManifest
    | GroundRecord
    | CandidateManifest
    | AnchorManifest
    | IndexPageRecord
)

AnchorRef: TypeAlias = StateRef[AnchorManifest]
CandidateRef: TypeAlias = StateRef[CandidateManifest]
EvaluationRef: TypeAlias = AnchorRef | CandidateRef
