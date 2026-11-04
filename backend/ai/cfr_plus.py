import numpy as np

from backend.ai.cfr import CFR


class CFRPlus(CFR):
    def _average_weight(self):
        return self.iteration

    def _after_regret_update(self):
        # Complete all history contributions before clipping, once per player pass.
        np.maximum(self.regrets, 0, out=self.regrets)
