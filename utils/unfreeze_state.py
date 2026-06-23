import torch


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


def _save_unfreeze_state(obj, path):
    """Save an UnfreezeState dictionary to the path SpeechBrain expects."""
    torch.save(obj.state_dict(), path)


def _load_unfreeze_state(obj, path, end_of_epoch):
    """Load an UnfreezeState dictionary; end_of_epoch is unused."""
    state = torch.load(path, map_location="cpu")
    obj.load_state_dict(state)
