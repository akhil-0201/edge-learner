"""
variant_tagger.py - Tag frames with panel / scene variant identifiers.

Used to annotate collected frames so the fine-tuner can perform
per-variant incremental learning without mixing domains.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class VariantTagger:
    """
    Assigns a *variant* label to each frame based on configurable rules.

    Rules are loaded from a YAML file that maps variant names to lists of
    keyword strings.  The tagger scans the frame *source* field (e.g. camera
    URI / file path) and returns the first matching variant name.

    Parameters
    ----------
    variants_config : str | Path
        Path to the variants YAML file (default: config/variants.yaml).
    default_variant : str
        Label returned when no rule matches.
    """

    def __init__(
        self,
        variants_config: str | Path = "config/variants.yaml",
        default_variant: str = "generic",
    ) -> None:
        self._default = default_variant
        self._rules: Dict[str, List[str]] = {}
        self._load(Path(variants_config))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tag(self, source: str) -> str:
        """
        Return the variant label for *source*.

        Parameters
        ----------
        source : str
            Camera URI, file path, or any identifier string.

        Returns
        -------
        str
            Matched variant name, or the default variant.
        """
        src_lower = source.lower()
        for variant, keywords in self._rules.items():
            if any(kw.lower() in src_lower for kw in keywords):
                return variant
        return self._default

    def variants(self) -> List[str]:
        """Return the list of known variant names (including default)."""
        known = list(self._rules.keys())
        if self._default not in known:
            known.append(self._default)
        return known

    def reload(self, variants_config: Optional[str | Path] = None) -> None:
        """Reload variant rules, optionally from a new path."""
        if variants_config is not None:
            self._load(Path(variants_config))
        logger.info("VariantTagger rules reloaded (%d variants)", len(self._rules))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> None:
        if not path.exists():
            logger.warning(
                "variants config not found at %s; using empty ruleset", path
            )
            self._rules = {}
            return

        with path.open() as fh:
            data = yaml.safe_load(fh) or {}

        variants = data.get("variants", {})
        if not isinstance(variants, dict):
            logger.error("variants.yaml must contain a top-level 'variants' mapping")
            self._rules = {}
            return

        self._rules = {
            name: (kws if isinstance(kws, list) else [str(kws)])
            for name, kws in variants.items()
        }
        logger.info(
            "VariantTagger loaded %d variant rules from %s", len(self._rules), path
        )
