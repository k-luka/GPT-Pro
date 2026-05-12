import torch


class DualOptimizer:
    def __init__(self, adam_opt, muon_opt):
        self.adam_opt = adam_opt
        self.muon_opt = muon_opt
        self.param_groups = adam_opt.param_groups + muon_opt.param_groups

    def get_adamw_params(self):
        return [p for group in self.adam_opt.param_groups for p in group["params"]]

    def zero_grad(self, set_to_none=True):
        self.adam_opt.zero_grad(set_to_none=set_to_none)
        self.muon_opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.adam_opt.step()
        self.muon_opt.step()

    def state_dict(self):
        return {
            "adam_opt": self.adam_opt.state_dict(),
            "muon_opt": self.muon_opt.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.adam_opt.load_state_dict(state_dict["adam_opt"])
        self.muon_opt.load_state_dict(state_dict["muon_opt"])
