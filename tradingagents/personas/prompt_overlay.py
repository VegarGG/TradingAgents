"""Persona system-prompt overlay helper.

Used by every agent factory (analysts, researchers, managers, trader, PM,
risk debators) to append the active persona's ``system_prompt_fragment``
to its base system prompt. Passthrough when ``persona is None``.
"""

from __future__ import annotations

from typing import Optional

from tradingagents.personas.loader import Persona


def apply_fragment(base_prompt: str, persona: Optional[Persona]) -> str:
    """Return ``base_prompt`` with the persona's fragment appended.

    No-op when ``persona is None`` or the persona's fragment is empty/blank.
    Inserts one blank line between the base and the fragment; strips any
    trailing whitespace at the end of the combined result.
    """
    if persona is None:
        return base_prompt
    fragment = (persona.system_prompt_fragment or "").strip()
    if not fragment:
        return base_prompt
    combined = f"{base_prompt.rstrip()}\n\n{fragment}"
    return combined.rstrip()
