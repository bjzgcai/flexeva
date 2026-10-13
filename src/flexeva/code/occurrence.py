"""Bounded dynamic loop positions for Code's source-occurrence identity.

The shared CFG index locates existing runtime loop-bind/predicate hooks. Their
first and subsequent visits distinguish iterations even when an emission is
skipped on some ranks. No runtime values or per-microstep journal are kept.
"""

from contextvars import ContextVar


class OccurrenceTracker:
    def __init__(self, analysis=None):
        self.paths = dict(analysis.loop_paths) if analysis is not None else {}
        self.ticks = dict(analysis.loop_ticks) if analysis is not None else {}
        self.descendants = {
            tick: {other for other in self.ticks if tick in self.paths.get(other, ()) and other != tick}
            for tick in self.ticks
        }
        self.positions = {}
        self.calls = ContextVar("iteration_calls", default=())

    def observe(self, occurrence):
        key = occurrence.site_id
        role = self.ticks.get(key)
        if role is None or (role == "loop-test" and occurrence.inputs.get("operation") != "predicate"):
            return
        scope = tuple(occurrence.inputs.get("scope_path", ()))
        positions = self.positions.setdefault(scope, {})
        positions[key] = positions.get(key, -1) + 1
        for child in self.descendants[key]:
            positions.pop(child, None)

    def path(self, site_id, scope):
        positions = self.positions.get(tuple(scope), {})
        calls = self.calls.get()
        inherited = calls[-1] if calls else ()
        return (*inherited, *(f"loop:{key}:{positions.get(key, -1)}"
                              for key in self.paths.get(site_id, ())))

    def enter_call(self, site, scope):
        self.calls.set((*self.calls.get(), self.path(site.get("site_id"), scope)))

    def leave_call(self):
        self.calls.set(self.calls.get()[:-1])

    def deactivate(self, scope):
        self.positions.pop(tuple(scope), None)
