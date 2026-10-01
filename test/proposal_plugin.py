"""A test-only proposer with no Optuna dependency."""
from crest.interfaces import OptimizerABC
from crest.pipeline_types import CandidateProposal, ExplicitRound

class ThirdProposer(OptimizerABC):
    rounds = []
    calls = []
    contexts = []
    def identity_config(self, config):
        return {"setting": config.get("setting", 1)}
    def initialize(self, context, config):
        type(self).contexts.append(context)
    def propose_round(self, history, budget):
        type(self).calls.append((history, budget))
        result = type(self).rounds.pop(0)
        if isinstance(result, list):
            return ExplicitRound(tuple(CandidateProposal(params) for params in result))
        return result
