"""Backend-neutral Code event writer for the registered RAS capture sink."""

from contextlib import contextmanager
from types import MappingProxyType

from .evidence import CodeContractError, encode


class CodeEvidenceDictionary:
    """Exact, readable structural evidence in the capture's shared Code pool.

    Core identity compares complete canonical bytes. References never stand for
    a truncated hash. Every capture emits its own reachable definitions; native
    occurrence slices retain the entire pool, including its dictionary records.
    """
    def __init__(self, sink):
        self.sink = sink
        self.definitions = {}

    def intern(self, kind, value):
        encoded = encode(value)
        coordinate = kind.encode("ascii") + b"\0" + encoded
        identity = self.definitions.get(coordinate)
        if identity is None:
            identity = self.sink.identity(0x434f444544494301, coordinate)
            self.definitions[coordinate] = identity
            # encoded is already canonical: do not traverse its value again.
            self.sink.code_dictionary(b'{"id":' + str(identity).encode("ascii")
                + b',"kind":' + encode(kind) + b',"value":' + encoded + b'}')
        return {"$": identity}

    def factor(self, value):
        if type(value) in (dict, MappingProxyType):
            return self.intern("map", {key: self.factor(item) for key, item in value.items()})
        if type(value) in (tuple, list):
            items = tuple(self.factor(item) for item in value)
            return self.intern("sequence", items) if len(items) > 4 else items
        if type(value) is str and len(value) > 48:
            return self.intern("value", value)
        return value


def decode_code_packet(packet, definitions):
    """Expand a v2 packet for diagnostics using its owning pool's definitions.

    ``definitions`` maps integer IDs to decoded code_dictionary records. This
    reader reconstructs the public CodeEvent.as_dict contract without requiring
    production capture to build and encode that full tree a second time.
    """
    def expand(value):
        if isinstance(value, dict) and set(value) == {"$"}:
            record = definitions[value["$"]]
            if record["kind"] == "map":
                return {key: expand(item) for key, item in record["value"].items()}
            return expand(record["value"])
        if isinstance(value, (list, tuple)):
            return [expand(item) for item in value]
        return value
    entry = expand(packet["entry"])
    inputs = {key: value for key, value in entry.items() if key not in packet["remove"]}
    inputs.update(expand(packet["updates"]))
    return {"schema": packet["event_schema"], "identity": expand(packet["identity"]),
        "kind": packet["kind"], "inputs": inputs, "actual": expand(packet["actual"]),
        "representation": expand(packet["representation"])}


class CodeCaptureWriter:
    def __init__(self, sink, workload, rank, dictionary=None):
        self.sink, self.workload, self.rank = sink, workload, rank
        self.dictionary = dictionary if dictionary is not None else CodeEvidenceDictionary(sink)
        self.group = workload.rank_plan.group_for(rank)
        self.stack, self.published = [], set()
        self.preparation_identity = encode(("invocation",))
        self.preparation = self.sink.identity(0x4d41594152454701,
            self.preparation_identity + int(rank).to_bytes(8, "little"))
        self.modeled_depth = 0
        self.ground_facts = {}
        self._event_ids, self._key_bytes, self._sites = {}, {}, {}
        self._entries, self._block_history = {}, {}

    def activate(self, owner):
        self.modeled_depth += int(owner == self.workload.registration.modeled_function)

    def deactivate(self, owner):
        self.modeled_depth -= int(owner == self.workload.registration.modeled_function)

    def event_id(self, key, rank=None):
        rank = self.rank if rank is None else rank
        coordinate = key, rank
        if coordinate not in self._event_ids:
            if key not in self._key_bytes:
                self._key_bytes[key] = encode(key)
            self._event_ids[coordinate] = self.sink.identity(0x4d41594152454701,
                self._key_bytes[key] + int(rank).to_bytes(8, "little"))
        return self._event_ids[coordinate]

    def _row(self, key, semantic, *, complete, trace, reusable, site, rank=None):
        rank = self.rank if rank is None else rank
        if site not in self._sites:
            encoded = encode(site)
            self._sites[site] = self.sink.identity(0x434f444503, encoded), encoded
        site_id, site_identity = self._sites[site]
        identity = self.event_id(key, rank)
        return {"id": identity, "rank": rank, "site_id": site_id,
                "site_identity": site_identity, "occurrence_identity": self._key_bytes[key],
                "semantic": semantic, "semantic_complete": complete,
                "reuse_allowed": reusable, "produces_trace": trace,
                "execution_region": self.event_id(key)}

    @contextmanager
    def invocation(self, context, bindings):
        self.sink.declare_region({"id": self.preparation, "rank": self.rank,
            "site_id": self.preparation, "semantic": encode({"preparation": self.rank}),
            "site_identity": self.preparation_identity, "occurrence_identity": self.preparation_identity,
            "semantic_complete": False, "reuse_allowed": False, "produces_trace": False})
        self.sink.begin_region(self.preparation, self.rank)
        try:
            yield
        except BaseException:
            try:
                self.sink.end_region()
            except BaseException:
                pass  # The failed native transaction is discarded by Core.
            raise
        else:
            self.sink.end_region()

    @contextmanager
    def auxiliary(self, name):
        """Capture lifecycle resource changes outside the modeled Code window."""
        coordinate = encode(("context", name, self.rank))
        identity = self.sink.identity(0x434f444506, coordinate)
        self.sink.declare_region({"id": identity, "rank": self.rank,
            "site_id": identity, "site_identity": coordinate, "occurrence_identity": coordinate,
            "semantic": coordinate, "semantic_complete": False,
            "reuse_allowed": False, "produces_trace": False})
        self.sink.begin_region(identity, self.rank)
        try:
            yield
        except BaseException:
            try: self.sink.end_region()
            except BaseException: pass
            raise
        else:
            self.sink.end_region()

    def observe_block(self, key, evidence):
        # Each observed fragment is factored once; retain only its compact ref.
        # Stable producer coordinates include self-definitions inside a block.
        self._block_history.setdefault(key, []).append(self.dictionary.factor(evidence))

    def begin(self, key, occurrence, evidence):
        binding = occurrence.inputs.get("emission_binding") or {}
        needs_ground = binding.get("operation_id") == "ground-consumer"
        entry = self.dictionary.factor(evidence)
        self._entries[key] = evidence, entry
        semantic = encode(("code-entry-v2", entry))
        for rank in self.group.active_ranks:
            row = self._row(key, semantic, rank=rank, complete=bool(evidence["inputs_complete"]),
                trace=self.workload.registration.modeled_function is None or self.modeled_depth > 0,
                reusable=bool(evidence["inputs_complete"]) and not needs_ground,
                site=occurrence.site_id)
            row["inputs"] = tuple(self.event_id(producer, rank) for producer, _ in evidence["producer_inputs"])
            self.sink.declare_region(row)
        self.sink.begin_region(self.event_id(key), self.rank)
        self.stack.append(key)

    def end(self, key, error):
        if not self.stack or self.stack.pop() != key:
            raise CodeContractError("native Code occurrence stack mismatch")
        try:
            self.sink.end_region()
        except BaseException:
            if error is None:
                raise

    def publish(self, event):
        identity = self.event_id(event.identity)
        references = tuple(self.event_id(producer) for producer, _ in event.inputs["producer_inputs"])
        standalone = event.kind == "block"
        history = None
        if standalone:
            entry = self.dictionary.factor(event.inputs)
            initial = event.inputs
            history = self.dictionary.intern("sequence", self._block_history.pop(event.identity, ()))
            semantic = encode(("code-entry-v2", entry, history))
            for rank in self.group.active_ranks:
                row = self._row(event.identity, semantic, trace=False, rank=rank,
                                complete=bool(event.inputs["inputs_complete"]),
                                reusable=event.representation.reusable, site=event.actual.site)
                row["inputs"] = tuple(self.event_id(producer, rank) for producer, _ in event.inputs["producer_inputs"])
                self.sink.declare_region(row)
            self.sink.begin_region(identity, self.rank)
        else:
            initial, entry = self._entries.pop(event.identity)
        updates = {key: value for key, value in event.inputs.items()
                   if key not in initial or value != initial[key]}
        packet = {"schema": "native-code-occurrence-v2", "event_schema": event.schema,
            "identity": self.dictionary.factor(event.identity), "kind": event.kind,
            "entry": entry, "updates": self.dictionary.factor(updates),
            "remove": tuple(key for key in initial if key not in event.inputs),
            "history": history,
            "actual": self.dictionary.factor({"representative": event.actual.representative,
                "sequence": event.actual.sequence, "site": event.actual.site,
                "occurrence": event.actual.occurrence, "execution_digest": event.actual.execution_digest}),
            "representation": self.dictionary.factor({"active_ranks": event.representation.active_ranks,
                "mappings": event.representation.mappings, "equivalence": event.representation.equivalence,
                "reusable": event.representation.reusable})}
        try:
            self.sink.code_occurrence(identity, encode(packet),
                                      tuple(self.ground_facts.pop(event.identity, ())), references)
        except BaseException:
            if standalone:
                try:
                    self.sink.end_region()
                except BaseException:
                    pass
            raise
        else:
            if standalone:
                self.sink.end_region()
        self.published.add(event.identity)
