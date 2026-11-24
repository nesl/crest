"""Shared decoded FER duplicate rejection for native Optuna samplers."""
from itertools import product
import json
import optuna
from .decode import decoded_architecture_identity
from crest.interfaces import OptimizerBudgetExhausted


class _UniqueFERMixin:
    """Keep the underlying sampler proposal; replace duplicates only.

    Supports CREST's sequential native Optuna runner. This is not an atomic
    reservation mechanism for concurrent workers sharing the same study.
    """

    def __init__(self, *, choices, fixed_resolution, **kwargs):
        super().__init__(**kwargs)
        self._fer_space = {k: optuna.distributions.CategoricalDistribution(v) for k, v in choices.items()}
        self._fer_resolution = fixed_resolution

    def infer_relative_search_space(self, study, trial):
        return dict(self._fer_space)

    def _identity(self, params):
        values = {k: params[k] for k in self._fer_space}
        values.setdefault("resolution", self._fer_resolution)
        return json.dumps(decoded_architecture_identity(values), sort_keys=True)

    def sample_relative(self, study, trial, search_space):
        seen = {
            self._identity(t.params) for t in study.get_trials(deepcopy=False)
            if t.number != trial.number and all(k in t.params for k in self._fer_space)
        }
        # Preserve the underlying Optuna proposal first.
        candidate = super().sample_relative(study, trial, search_space)
        for name, distribution in search_space.items():
            if name not in candidate:
                candidate[name] = super().sample_independent(study, trial, name, distribution)
        if self._identity(candidate) not in seen:
            return candidate
        # Reuse the sampler's independent sampling for duplicate refill.
        for _ in range(64):
            candidate = {name: super(_UniqueFERMixin, self).sample_independent(study, trial, name, dist)
                         for name, dist in search_space.items()}
            if self._identity(candidate) not in seen:
                return candidate
        # Bounded random refill cannot hang near exhaustion. Enumerate only then.
        names = list(search_space)
        for values in product(*(search_space[k].choices for k in names)):
            candidate = dict(zip(names, values))
            if self._identity(candidate) not in seen:
                return candidate
        raise OptimizerBudgetExhausted("All decoded FER architectures in the configured space have been attempted.")


class UniqueFERNSGAIISampler(_UniqueFERMixin, optuna.samplers.NSGAIISampler):
    """NSGA-II proposals with shared decoded-architecture uniqueness."""


class UniqueFERTPESampler(_UniqueFERMixin, optuna.samplers.TPESampler):
    """TPE proposals with shared decoded-architecture uniqueness."""
