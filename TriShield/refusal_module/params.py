class Params:
    def __init__(self, layer):
        self.layer = layer
        self.deltas = None
        self.enable_hook = False

    def set_deltas(self, deltas):
        self.deltas = deltas
