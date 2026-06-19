# utils/unfreeze_state.py

class UnfreezeState:
    def __init__(self, unfrozen_count=0, phase="heads_only"):
        self.unfrozen_count = int(unfrozen_count)
        self.phase = str(phase)
        self._restored_from_ckpt = False

    def state_dict(self):
        return {
            "unfrozen_count": int(self.unfrozen_count),
            "phase": str(self.phase),
        }

    def load_state_dict(self, state):
        self.unfrozen_count = int(state.get("unfrozen_count", 0))
        self.phase = str(state.get("phase", "heads_only"))
        self._restored_from_ckpt = True
