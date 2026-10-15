# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Persistent, conservative spend accounting for one optimizer writer."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any

from .provider import ProviderError, SpendLimitExceeded


class SpendInputLimitExceeded(ProviderError):
    """The conservative prompt bound exceeds the configured input limit."""


class SpendAccountingError(ProviderError):
    """Accounting is invalid or the provider exceeded its reserved token limits."""


def _decimal(value: Any, name: str) -> Decimal:
    """Parse a finite, nonnegative decimal without binary float arithmetic.

    Parameters
    ----------
    value : object
        Configured number or saved decimal string.
    name : str
        Setting name included in validation errors.

    Returns
    -------
    decimal.Decimal
        Exact nonnegative amount.
    """
    try:
        if isinstance(value, bool):
            raise ValueError
        result = Decimal(str(value))
        if not result.is_finite() or result < 0:
            raise ValueError
        return result
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise SpendAccountingError(f"{name} must be a finite nonnegative number.") from exc


def _text(value: Decimal) -> str:
    """Return an exact JSON-safe decimal with stable equivalent spellings.

    Parameters
    ----------
    value : decimal.Decimal
        Finite amount to serialize.

    Returns
    -------
    str
        Fixed-point representation without redundant fractional zeros.
    """
    result = format(value, "f")
    return result.rstrip("0").rstrip(".") if "." in result else result


def _sum_amounts(*amounts: Decimal) -> Decimal:
    """Add monetary amounts without rounding significant decimal digits.

    Parameters
    ----------
    *amounts : decimal.Decimal
        Finite signed amounts, including any release of a prior reservation.

    Returns
    -------
    decimal.Decimal
        Exact sum with precision sized for all integer and fractional digits.
    """
    with localcontext() as context:
        context.prec = max(28, max(value.adjusted() for value in amounts)
                           - min(value.as_tuple().exponent for value in amounts)
                           + len(str(len(amounts))) + 3)
        return sum(amounts, Decimal(0))


class SpendTracker:
    """Reserve worst-case cost before each HTTP attempt and persist actual usage.

    Parameters
    ----------
    root : pathlib.Path
        Existing optimizer ledger directory, or a directory to create. There must
        be only one writer for this directory.
    config : Mapping
        Flat provider settings containing ``max_usd``, both per-million token
        prices, nonempty ``pricing_source``, positive integer input/output token
        limits, and model, provider, base URL, and output-token parameter names.

    Notes
    -----
    Unresolved attempts remain charged at their full reservation after a crash
    or transport failure. The monetary bound relies on the configured prices and
    on the provider honoring its output limit. An observed token overrun is
    charged and permanently stops further attempts from this state file.
    """

    def __init__(self, root: Path, config: Mapping[str, Any]) -> None:
        """Create or restore accounting bound to the configured pricing policy.

        Parameters
        ----------
        root : pathlib.Path
            Single-writer optimizer ledger directory.
        config : Mapping
            Provider identity, positive prices, provenance, token limits and cap.

        Raises
        ------
        SpendAccountingError
            Configuration or saved accounting is invalid or incompatible.
        SpendLimitExceeded
            The new cap is below already reserved or charged cost.
        """
        self.root = Path(root)
        self.path = self.root / "spend_state.json"
        self.max_usd = _decimal(config.get("max_usd"), "max_usd")
        self.input_price = _decimal(config.get("input_usd_per_million_tokens"), "input_usd_per_million_tokens")
        self.output_price = _decimal(config.get("output_usd_per_million_tokens"), "output_usd_per_million_tokens")
        if self.input_price == 0 or self.output_price == 0:
            raise SpendAccountingError("Input and output token prices must be strictly positive.")
        self.max_input_tokens = self._token_limit(config, "max_input_tokens")
        self.max_output_tokens = self._token_limit(config, "max_output_tokens")
        binding: dict[str, Any] = {
            "input_usd_per_million_tokens": _text(self.input_price),
            "output_usd_per_million_tokens": _text(self.output_price),
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
        }
        for name in ("pricing_source", "model", "provider", "base_url", "output_token_parameter"):
            value = config.get(name)
            if not isinstance(value, str) or not value.strip():
                raise SpendAccountingError(f"{name} must be a nonempty string.")
            binding[name] = value.strip().rstrip("/") if name == "base_url" else value.strip()
        # Tier selection can change billing; preserve legacy bindings when absent.
        if config.get("service_tier") is not None:
            binding["service_tier"] = config["service_tier"]
        self.reservation_usd = self._cost(self.max_input_tokens, self.max_output_tokens)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                state = json.loads(self.path.read_text(encoding="utf-8"))
                self._validate_state(state, binding)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                raise SpendAccountingError("Cannot restore spend_state.json; refusing to reset accounting.") from exc
            if self.max_usd < _decimal(state["charged_usd"], "charged_usd"):
                raise SpendLimitExceeded("max_usd cannot be reduced below already charged/reserved cost.")
            state["max_usd"] = _text(self.max_usd)
        else:
            if any(next((self.root / name).glob("*.request.json"), None) is not None
                   for name in ("requests", "memory/requests")):
                raise SpendAccountingError(
                    "Existing request ledger has no spend state; prior cost cannot be reconstructed safely."
                )
            state = {
                "schema_version": 1,
                "binding": binding,
                "max_usd": _text(self.max_usd),
                "charged_usd": "0",
                "next_reservation_id": 1,
                "halted": False,
                "reservations": {},
            }
        self._persist(state)
        self._state = state

    @staticmethod
    def _token_limit(config: Mapping[str, Any], name: str) -> int:
        """Require integer limits, excluding booleans and implicit coercions.

        Parameters
        ----------
        config : Mapping
            Flat provider settings.
        name : str
            Input or output token-limit setting.

        Returns
        -------
        int
            Positive token ceiling.
        """
        value = config.get(name)
        if type(value) is not int or value < 1:
            raise SpendAccountingError(f"{name} must be a positive integer.")
        return value

    def _cost(self, prompt_tokens: int, completion_tokens: int) -> Decimal:
        """Compute exact configured cost, allowing more than default precision.

        Parameters
        ----------
        prompt_tokens : int
            Number of input tokens.
        completion_tokens : int
            Number of generated tokens.

        Returns
        -------
        decimal.Decimal
            Configured USD cost for both token counts.
        """
        with localcontext() as context:
            context.prec = max(28, len(str(prompt_tokens)) + len(str(completion_tokens))
                               + len(self.input_price.as_tuple().digits)
                               + len(self.output_price.as_tuple().digits)
                               + abs(self.input_price.as_tuple().exponent)
                               + abs(self.output_price.as_tuple().exponent) + 16)
            return (self.input_price * prompt_tokens + self.output_price * completion_tokens) / Decimal(1_000_000)

    def _validate_state(self, state: dict[str, Any], binding: dict[str, Any]) -> None:
        """Reject incompatible or inconsistent persisted accounting.

        Parameters
        ----------
        state : dict
            Decoded saved accounting state.
        binding : dict
            Current normalized immutable provider and pricing settings.

        Raises
        ------
        SpendAccountingError
            State violates the binding or its reservation totals disagree.
        """
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise SpendAccountingError("Unsupported spend state schema.")
        if state.get("binding") != binding:
            raise SpendAccountingError("Spend settings changed; model, prices, limits, and pricing provenance must match saved state.")
        reservations = state["reservations"]
        if not isinstance(reservations, dict) or type(state["halted"]) is not bool:
            raise SpendAccountingError("Invalid spend state records.")
        next_id = state["next_reservation_id"]
        if type(next_id) is not int or next_id < 1:
            raise SpendAccountingError("Invalid spend reservation sequence.")
        total = Decimal(0)
        for reservation_id, record in reservations.items():
            if not reservation_id.isdecimal() or not 1 <= int(reservation_id) < next_id:
                raise SpendAccountingError("Invalid spend reservation identifier.")
            if not isinstance(record, dict):
                raise SpendAccountingError("Invalid spend reservation record.")
            if record["status"] not in {"reserved", "settled", "unknown", "exceeded"}:
                raise SpendAccountingError("Invalid spend reservation status.")
            reserved = _decimal(record["reserved_usd"], "reserved_usd")
            charged = _decimal(record["charged_usd"], "charged_usd")
            if reserved != self.reservation_usd or (record["status"] in {"reserved", "unknown"} and charged != reserved):
                raise SpendAccountingError("Invalid spend reservation cost.")
            total = _sum_amounts(total, charged)
        if total != _decimal(state["charged_usd"], "charged_usd"):
            raise SpendAccountingError("Saved spend total does not equal reservation charges.")
        _decimal(state["max_usd"], "max_usd")

    def _persist(self, state: dict[str, Any]) -> None:
        """Atomically replace and fsync state before allowing a network attempt.

        Parameters
        ----------
        state : dict
            Complete next accounting state.

        Raises
        ------
        SpendAccountingError
            The state could not be durably written.
        """
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.root,
                                             prefix=".spend-", suffix=".tmp", delete=False) as handle:
                temporary_path = Path(handle.name)
                json.dump(state, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
            directory_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as exc:
            raise SpendAccountingError("Cannot persist spend reservation; provider attempt stopped.") from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def reserve(self, request_payload: dict[str, Any]) -> str:
        """Persist a full-cost reservation before one HTTP attempt.

        Parameters
        ----------
        request_payload : dict
            Chat-completions payload containing the serialized messages.

        Returns
        -------
        str
            Persistent reservation identifier for settlement after this attempt.

        Raises
        ------
        SpendInputLimitExceeded
            Serialized UTF-8 message bytes plus 1024 framing tokens exceed the
            input allowance. This conservative bound intentionally overcounts.
        SpendLimitExceeded
            Remaining cap cannot cover both configured token ceilings.
        SpendAccountingError
            State is halted or messages cannot be serialized safely.
        """
        if self._state["halted"]:
            raise SpendAccountingError("Provider previously exceeded reserved token limits; spend accounting is halted.")
        try:
            if "input" in request_payload:
                # Bound the entire Responses payload, including formatting/options.
                messages = request_payload
            else:
                messages = request_payload["messages"]
                if not isinstance(messages, list):
                    raise TypeError("messages must be a list")
            input_bound = len(json.dumps(messages, separators=(",", ":"), allow_nan=False).encode("utf-8")) + 1024
        except (KeyError, TypeError, ValueError, UnicodeError) as exc:
            raise SpendAccountingError("Cannot bound serialized request messages.") from exc
        if input_bound > self.max_input_tokens:
            raise SpendInputLimitExceeded(f"Conservative input bound {input_bound} exceeds max_input_tokens={self.max_input_tokens}.")
        total = _sum_amounts(_decimal(self._state["charged_usd"], "charged_usd"), self.reservation_usd)
        if total > self.max_usd:
            raise SpendLimitExceeded(f"LLM spend cap cannot cover the next full reservation of ${_text(self.reservation_usd)}.")
        state = copy.deepcopy(self._state)
        reservation_id = str(state["next_reservation_id"])
        state["next_reservation_id"] += 1
        state["reservations"][reservation_id] = {
            "status": "reserved", "reserved_usd": _text(self.reservation_usd),
            "charged_usd": _text(self.reservation_usd), "usage_known": False,
            **({"wire_payload": request_payload} if "input" in request_payload else {}),
        }
        state["charged_usd"] = _text(total)
        self._persist(state)
        self._state = state
        return reservation_id

    def settle(self, reservation_id: str, usage: dict[str, Any] | None) -> dict[str, Any]:
        """Charge valid actual token usage or retain the entire reservation.

        Parameters
        ----------
        reservation_id : str
            Identifier returned before the corresponding HTTP attempt.
        usage : dict or None
            Provider usage. Both ``prompt_tokens`` and ``completion_tokens`` must
            be nonnegative integers; any missing or malformed value keeps the
            full reservation charged.

        Returns
        -------
        dict
            JSON-safe accounting with exact decimal USD strings. Repeated
            settlement does not charge or release funds a second time.

        Raises
        ------
        SpendAccountingError
            The identifier is unknown or actual usage exceeds either token
            ceiling. Overrun charges are persisted before raising.
        """
        if reservation_id not in self._state["reservations"]:
            raise SpendAccountingError(f"Unknown spend reservation {reservation_id!r}.")
        record = self._state["reservations"][reservation_id]
        if record["status"] == "reserved":
            state = copy.deepcopy(self._state)
            record = state["reservations"][reservation_id]
            valid = isinstance(usage, dict) and all(
                type(usage.get(name)) is int and usage[name] >= 0
                for name in ("prompt_tokens", "completion_tokens")
            )
            record["usage_known"] = valid
            exceeded = False
            if valid:
                prompt, completion = usage["prompt_tokens"], usage["completion_tokens"]
                actual = self._cost(prompt, completion)
                state["charged_usd"] = _text(_sum_amounts(
                    _decimal(state["charged_usd"], "charged_usd"),
                    _decimal(record["charged_usd"], "charged_usd").copy_negate(), actual,
                ))
                record.update(prompt_tokens=prompt, completion_tokens=completion, charged_usd=_text(actual))
                if "input_tokens" in usage:
                    record.update(cached_tokens=usage.get("cached_tokens", 0),
                                  cache_write_tokens=usage.get("cache_write_tokens", 0))
                exceeded = prompt > self.max_input_tokens or completion > self.max_output_tokens
                record["status"] = "exceeded" if exceeded else "settled"
                state["halted"] = exceeded
            else:
                record["status"] = "unknown"
            self._persist(state)
            self._state = state
        if record["status"] == "exceeded":
            raise SpendAccountingError("Provider usage exceeded reserved token limits; actual cost was charged and further attempts are halted.")
        return {"reservation_id": reservation_id, **record,
                "total_charged_usd": self._state["charged_usd"], "max_usd": _text(self.max_usd)}
