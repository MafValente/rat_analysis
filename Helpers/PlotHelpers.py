"""Small, notebook-facing helpers for adjusting existing Matplotlib figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from matplotlib.figure import Figure


def _resolve_path(namespace: Mapping[str, Any], path: str) -> Any:
    """Resolve ``name.mapping_key.attribute`` paths from a notebook namespace."""
    parts = path.split(".")
    if not parts or parts[0] not in namespace:
        raise KeyError(path)
    value: Any = namespace[parts[0]]
    for part in parts[1:]:
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        elif hasattr(value, part):
            value = getattr(value, part)
        else:
            raise KeyError(path)
    return value


def _iter_figures(value: Any, seen: set[int] | None = None):
    """Yield figures contained in a figure, mapping, or short sequence."""
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)

    if isinstance(value, Figure):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_figures(child, seen)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _iter_figures(child, seen)


def _axis_matches(ax, axis_index: int, selector: str | int) -> bool:
    if isinstance(selector, int):
        return axis_index == selector

    selector_text = str(selector).strip()
    if selector_text.lower() in {"all", "*"}:
        return True
    if selector_text.isdigit():
        return axis_index == int(selector_text)

    fields = {
        "title": ax.get_title(),
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
    }
    if ":" in selector_text:
        field, wanted = selector_text.split(":", 1)
        field = field.strip().lower()
        if field in fields:
            return wanted.strip().lower() in fields[field].lower()

    searchable = " ".join(fields.values()).lower()
    return selector_text.lower() in searchable


def _validate_limit(limit: Any, name: str):
    if limit is None:
        return None
    if not isinstance(limit, Sequence) or isinstance(limit, (str, bytes)) or len(limit) != 2:
        raise ValueError(f"{name} must be None or a two-value sequence, got {limit!r}.")
    return tuple(limit)


def _normalise_rules(rules: Mapping[Any, Any]) -> Mapping[Any, Mapping[str, Any]]:
    if "xlim" in rules or "ylim" in rules:
        return {"all": rules}
    return rules


def apply_axis_limit_config(
    namespace: Mapping[str, Any],
    config: Mapping[str, Mapping[Any, Any]],
    *,
    redisplay: bool = True,
    strict: bool = False,
) -> int:
    """Apply editable axis-limit rules to figures already created by a notebook.

    Top-level keys are notebook variable paths, such as ``"fig_daily"`` or
    ``"out.figures"``. Within each target, selectors may be ``"all"``, an
    integer axis index, or text matched against an axis title/x/y label. Prefix
    text selectors with ``title:``, ``xlabel:``, or ``ylabel:`` to restrict the
    match. A limit of ``None`` preserves the plot's existing behaviour.

    Returns the number of distinct figures changed.
    """
    changed_figures: list[Figure] = []

    for path, raw_rules in config.items():
        try:
            target = _resolve_path(namespace, path)
        except KeyError:
            if strict:
                raise
            continue

        if not isinstance(raw_rules, Mapping):
            raise TypeError(f"Axis-limit rules for {path!r} must be a mapping.")
        rules = _normalise_rules(raw_rules)

        for fig in _iter_figures(target):
            figure_changed = False
            for selector, settings in rules.items():
                if not isinstance(settings, Mapping):
                    raise TypeError(
                        f"Settings for {path!r}, selector {selector!r}, must be a mapping."
                    )
                xlim = _validate_limit(settings.get("xlim"), "xlim")
                ylim = _validate_limit(settings.get("ylim"), "ylim")
                if xlim is None and ylim is None:
                    continue
                for axis_index, ax in enumerate(fig.axes):
                    if not _axis_matches(ax, axis_index, selector):
                        continue
                    if xlim is not None:
                        ax.set_xlim(xlim)
                    if ylim is not None:
                        ax.set_ylim(ylim)
                    figure_changed = True
            if figure_changed:
                fig.canvas.draw_idle()
                if fig not in changed_figures:
                    changed_figures.append(fig)

    if redisplay and changed_figures:
        try:
            from IPython.display import display
        except ImportError:
            pass
        else:
            for fig in changed_figures:
                display(fig)

    return len(changed_figures)


def describe_figure_axes(namespace: Mapping[str, Any], path: str) -> list[dict[str, Any]]:
    """Return axis indices and labels to help choose selectors in a notebook."""
    target = _resolve_path(namespace, path)
    rows = []
    for figure_index, fig in enumerate(_iter_figures(target)):
        for axis_index, ax in enumerate(fig.axes):
            rows.append(
                {
                    "figure": figure_index,
                    "axis": axis_index,
                    "title": ax.get_title(),
                    "xlabel": ax.get_xlabel(),
                    "ylabel": ax.get_ylabel(),
                }
            )
    return rows
