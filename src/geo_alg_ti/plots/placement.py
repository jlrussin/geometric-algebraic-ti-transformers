"""Short texts placed inside an axes, clear of everything drawn there.

A text drawn for placement is marked with the point it starts from (`mark`). Once the
figure's layout is final, `place_texts` moves each marked text of an axes the least, in
one-point steps, until it clears every line, marker, band, legend and other text by `clear_pt`
and stays inside the axes by as much; a text marked with the color of its own line may sit on
that line's band and must sit nearer that line than any other marked line, where another line
coincides with its own counting as its own.
`audit_texts` reports each marked text's clearances. A text that no position within
`search_pt` (along each axis) clears takes, among the positions there inside the axes, the one
outside other bands and the legend and nearer its own line with the most clearance, or stays
at its anchor when none fits inside; the audit names it either way. Placement is
deterministic: the text with the fewest clear positions (with the other marked texts set
aside) is placed first, ties in the order they were drawn, and each sees the ones already
placed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.colors import to_hex
from matplotlib.path import Path
from matplotlib.text import Text

# The clearance, in points, from a placed text to anything drawn and to the axes edges, and
# the farthest, in points along each axis, a text may move from where it starts.
CLEAR_PT: float = 2.0
SEARCH_PT: float = 40.0
GID_PREFIX: str = "placed_text:"
# Candidate positions are tested in batches of this many.
_BATCH: int = 2048


def mark(text: Text, *, key: str, anchor: tuple[float, float], own_color: str | None = None) -> Text:
    """Mark `text` for placement from `anchor` (data units) under `key`, the name its records
    and audit use; `own_color` is the color of the line the text belongs to. The text must be
    drawn in its axes' data coordinates."""
    if text.axes is None or text.get_transform() is not text.axes.transData:
        raise ValueError(f"{text.get_text()!r}: a marked text is drawn in its axes' data coordinates")
    text.set_gid(GID_PREFIX + key)
    text.set_in_layout(False)
    text.placement_anchor = (float(anchor[0]), float(anchor[1]))
    text.placement_own_color = None if own_color is None else to_hex(own_color)
    return text


def marked_texts(ax: Axes) -> list[Text]:
    """The texts of `ax` marked for placement, in drawing order."""
    return [t for t in ax.texts if isinstance(t.get_gid(), str) and t.get_gid().startswith(GID_PREFIX)]


def _sample(xy: np.ndarray, radius: float, out: list) -> None:
    """Points at most 1.5 display units apart along a polyline, each with `radius`."""
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(2, int(np.ceil(np.hypot(*(b - a)) / 1.5)) + 1)
        for t in np.linspace(0.0, 1.0, n):
            out.append((*(a + (b - a) * t), radius))


def _scene(ax: Axes, renderer: object, skip: set, own_color: str | None = None) -> tuple[np.ndarray, list[Path], dict[str, np.ndarray]]:
    """What a text must clear, in display units: sampled points with radii, the closed areas
    (bands, the legend) a text may not enter, and the points of each marked line by color. A
    band in `own_color` (the text's own line's) is neither an area nor an edge to clear."""
    points: list = []
    areas: list[Path] = []
    lines_by_color: dict[str, list] = {}
    k = ax.figure.dpi / 72.0
    for line in ax.get_lines():
        if not line.get_visible():
            continue
        xy = line.get_transform().transform(np.column_stack(line.get_data()))
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) == 0:
            continue
        has_marker = line.get_marker() not in (None, "None", "", " ")
        if line.get_linestyle() not in ("None", "", " ", None):
            _sample(xy, line.get_linewidth() * k / 2, points)
        if has_marker:
            radius = (line.get_markersize() + line.get_markeredgewidth()) * k / 2
            points += [(x, y, radius) for x, y in xy]
            own: list = [(x, y, 0.0) for x, y in xy]
            _sample(xy, 0.0, own)
            lines_by_color.setdefault(to_hex(line.get_color()), []).extend(own)
    for collection in ax.collections:
        if not collection.get_visible():
            continue
        if isinstance(collection, LineCollection):
            width = float(np.max(collection.get_linewidths())) * k / 2
            for segment in collection.get_segments():
                _sample(collection.get_transform().transform(segment), width, points)
        elif isinstance(collection, PathCollection):
            edge = float(np.max(collection.get_linewidths())) if len(collection.get_linewidths()) else 0.0
            radius = (np.sqrt(np.max(collection.get_sizes())) + edge) * k / 2
            offsets = collection.get_offset_transform().transform(collection.get_offsets())
            points += [(x, y, radius) for x, y in offsets]
        elif isinstance(collection, PolyCollection):
            faces = collection.get_facecolors()
            if own_color is not None and len(faces) and to_hex(faces[0], keep_alpha=False) == own_color:
                continue
            transform = collection.get_transform()
            for path in collection.get_paths():
                outline = transform.transform(path.vertices)
                areas.append(Path(outline))
                _sample(np.vstack([outline, outline[:1]]), 0.0, points)
    legend = ax.get_legend()
    if legend is not None and legend.get_visible():
        box = legend.get_window_extent(renderer)
        corners = np.array([(box.x0, box.y0), (box.x1, box.y0), (box.x1, box.y1), (box.x0, box.y1), (box.x0, box.y0)])
        areas.append(Path(corners))
        _sample(corners, 0.0, points)
    for text in ax.texts:
        if text in skip or not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        _sample(np.array([(box.x0, box.y0), (box.x1, box.y0), (box.x1, box.y1), (box.x0, box.y1), (box.x0, box.y0)]), 0.0, points)
    by_color = {color: np.asarray(pts, dtype=float).reshape(-1, 3)[:, :2] for color, pts in lines_by_color.items()}
    return np.asarray(points, dtype=float).reshape(-1, 3), areas, by_color


def _search_order(search_pt: float) -> np.ndarray:
    """Offsets in points, one-point steps, nearest first; ties prefer up, then left."""
    reach = float(np.floor(search_pt))
    steps = np.arange(-reach, reach + 1e-9, 1.0)
    dx, dy = (a.ravel() for a in np.meshgrid(steps, steps, indexing="ij"))
    order = np.lexsort((dx, -dy, dx**2 + dy**2))
    return np.column_stack((dx[order], dy[order]))


def _gaps(lo_x: np.ndarray, hi_x: np.ndarray, lo_y: np.ndarray, hi_y: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Clearance from each candidate box (rows) to each point's edge (columns)."""
    gx = np.maximum(np.maximum(lo_x[:, None] - points[None, :, 0], 0), points[None, :, 0] - hi_x[:, None])
    gy = np.maximum(np.maximum(lo_y[:, None] - points[None, :, 1], 0), points[None, :, 1] - hi_y[:, None])
    return np.hypot(gx, gy) - points[None, :, 2]


def _box_probes(lo_x: np.ndarray, hi_x: np.ndarray, lo_y: np.ndarray, hi_y: np.ndarray) -> np.ndarray:
    """Nine points on each candidate box (corners, edge midpoints, center), shape (n, 9, 2)."""
    xs = np.stack([lo_x, (lo_x + hi_x) / 2, hi_x], axis=1)
    ys = np.stack([lo_y, (lo_y + hi_y) / 2, hi_y], axis=1)
    return np.stack([np.repeat(xs, 3, axis=1), np.tile(ys, (1, 3))], axis=2)


def _enters(areas: Sequence[Path], probes: np.ndarray) -> np.ndarray:
    """Whether any probe of each candidate box lies inside any closed area."""
    hit = np.zeros(probes.shape[0], dtype=bool)
    flat = probes.reshape(-1, 2)
    for area in areas:
        hit |= area.contains_points(flat).reshape(probes.shape[:2]).any(axis=1)
    return hit


def _nearest(lo_x: np.ndarray, hi_x: np.ndarray, lo_y: np.ndarray, hi_y: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Distance from each candidate box to the nearest of `pts` (infinite when there are none)."""
    if len(pts) == 0:
        return np.full(len(lo_x), np.inf)
    return _gaps(lo_x, hi_x, lo_y, hi_y, np.column_stack([pts, np.zeros(len(pts))])).min(axis=1)


def _ownership(text: Text, lines_by_color: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    """The points of the text's own marked line and of every other marked line, leaving out
    where another line coincides with its own (the points the two share); None when the text
    has no own line among them."""
    own = getattr(text, "placement_own_color", None)
    if own is None or own not in lines_by_color:
        return None
    shared = {tuple(p) for p in lines_by_color[own]}
    others = [pts[[tuple(p) not in shared for p in pts]] for color, pts in lines_by_color.items() if color != own]
    others = [pts for pts in others if len(pts)]
    return lines_by_color[own], (np.vstack(others) if others else np.empty((0, 2)))


def _owned(ownership: tuple[np.ndarray, np.ndarray] | None, lo_x, hi_x, lo_y, hi_y) -> np.ndarray:
    """Whether each candidate box is nearer the text's own line than any other marked line, the
    points as `_ownership` gives them."""
    if ownership is None:
        return np.ones(len(lo_x), dtype=bool)
    mine, theirs = ownership
    return _nearest(lo_x, hi_x, lo_y, hi_y, mine) < _nearest(lo_x, hi_x, lo_y, hi_y, theirs)


def _candidates(ax: Axes, text: Text, renderer: object, offsets: np.ndarray, clear: float, skip: set) -> tuple[np.ndarray, np.ndarray]:
    """The offsets at which `text` clears everything drawn but the texts in `skip`, nearest
    first, and the least crowded offset inside the axes (its index; -1 when none fits)."""
    frame = ax.get_window_extent(renderer)
    points, areas, lines_by_color = _scene(ax, renderer, skip | {text}, text.placement_own_color)
    ownership = _ownership(text, lines_by_color)
    box = text.get_window_extent(renderer)
    lo_x, hi_x = box.x0 + offsets[:, 0], box.x1 + offsets[:, 0]
    lo_y, hi_y = box.y0 + offsets[:, 1], box.y1 + offsets[:, 1]
    inside = (lo_x >= frame.x0 + clear) & (hi_x <= frame.x1 - clear) & (lo_y >= frame.y0 + clear) & (hi_y <= frame.y1 - clear)
    candidates = np.flatnonzero(inside)
    keep, best, best_score = [], -1, -np.inf
    for start in range(0, len(candidates), _BATCH):
        idx = candidates[start : start + _BATCH]
        gap = _gaps(lo_x[idx], hi_x[idx], lo_y[idx], hi_y[idx], points).min(axis=1) if len(points) else np.full(len(idx), np.inf)
        free = ~_enters(areas, _box_probes(lo_x[idx], hi_x[idx], lo_y[idx], hi_y[idx])) if areas else np.ones(len(idx), dtype=bool)
        owned = _owned(ownership, lo_x[idx], hi_x[idx], lo_y[idx], hi_y[idx])
        keep.append(idx[(gap >= clear) & free & owned])
        score = np.where(free & owned, gap, gap - 1e6)
        if len(idx) and score.max() > best_score:
            best, best_score = int(idx[np.argmax(score)]), float(score.max())
    return (np.concatenate(keep) if keep else np.array([], dtype=int)), best


def place_texts(ax: Axes, *, clear_pt: float = CLEAR_PT, search_pt: float = SEARCH_PT, check: bool = False) -> list[dict]:
    """Move each marked text of `ax` the least from its anchor until it is clear (see the module).

    Call it once the figure's layout is final; it restarts from the anchors, so calling it
    again gives the same result. A text that no position within `search_pt` clears falls back
    as the module describes (`audit_texts` names it); with `check`, the audit runs on the placed
    texts and any problem it finds raises. Returns each text's record, in drawing order.
    """
    fig = ax.figure
    texts = marked_texts(ax)
    for text in texts:
        text.set_position(text.placement_anchor)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    k = fig.dpi / 72.0
    clear = clear_pt * k
    offsets = _search_order(search_pt) * k
    others = set(texts)
    room = [len(_candidates(ax, text, renderer, offsets, clear, others)[0]) for text in texts]
    placed: set = set()
    for index in sorted(range(len(texts)), key=lambda i: (room[i], i)):
        text = texts[index]
        unplaced = others - placed - {text}
        ok, least_crowded = _candidates(ax, text, renderer, offsets, clear, unplaced)
        chosen = ok.min() if len(ok) else least_crowded
        if chosen < 0:
            placed.add(text)
            continue
        x0, y0 = ax.transData.transform(text.get_position())
        text.set_position(tuple(float(v) for v in ax.transData.inverted().transform((x0 + offsets[chosen, 0], y0 + offsets[chosen, 1]))))
        placed.add(text)
    if check:
        found = problems(_audit(ax, renderer, clear_pt=clear_pt))
        if found:
            raise ValueError(f"the marked texts could not all be placed clear: {'; '.join(found)}")
    return [_record(ax, text) for text in texts]


def _key(text: Text) -> str:
    return text.get_gid()[len(GID_PREFIX) :]


def _record(ax: Axes, text: Text) -> dict:
    """A placed text's key, string, color, position and anchor (data units), and its offset in
    points."""
    start = ax.transData.transform(text.placement_anchor)
    now = ax.transData.transform(text.get_position())
    k = ax.figure.dpi / 72.0
    return {
        "key": _key(text),
        "text": text.get_text(),
        "color": to_hex(text.get_color()),
        "position": [round(float(v), 4) for v in text.get_position()],
        "anchor": [round(float(v), 4) for v in text.placement_anchor],
        "offset_pt": [round(float(v), 2) for v in (now - start) / k],
        "ha": text.get_horizontalalignment(),
        "va": text.get_verticalalignment(),
    }


def audit_texts(ax: Axes, *, clear_pt: float = CLEAR_PT) -> dict:
    """Each marked text's clearance (points) to the nearest thing drawn and to the axes edges,
    and the texts that overlap something (`overlaps`), sit closer than `clear_pt` (`crowded`),
    reach outside the axes (`outside`), enter another line's band or the legend
    (`inside_area`), or sit nearer another marked line than their own (`misowned`), where
    another line coincides with its own counting as its own."""
    fig = ax.figure
    fig.canvas.draw()
    return _audit(ax, fig.canvas.get_renderer(), clear_pt=clear_pt)


def _audit(ax: Axes, renderer: object, *, clear_pt: float) -> dict:
    """`audit_texts` measured with `renderer`, without drawing again."""
    k = ax.figure.dpi / 72.0
    frame = ax.get_window_extent(renderer)
    out: dict = {"clearance_pt": {}, "edge_clearance_pt": {}, "overlaps": [], "crowded": [], "outside": [], "inside_area": [], "misowned": []}
    for text in marked_texts(ax):
        name = _key(text)
        points, areas, lines_by_color = _scene(ax, renderer, {text}, text.placement_own_color)
        box = text.get_window_extent(renderer)
        lo_x, hi_x, lo_y, hi_y = (np.array([v]) for v in (box.x0, box.x1, box.y0, box.y1))
        gap = float(_gaps(lo_x, hi_x, lo_y, hi_y, points).min()) / k if len(points) else float("inf")
        edge = float(min(box.x0 - frame.x0, frame.x1 - box.x1, box.y0 - frame.y0, frame.y1 - box.y1)) / k
        out["clearance_pt"][name] = round(gap, 3)
        out["edge_clearance_pt"][name] = round(edge, 3)
        if gap < 0:
            out["overlaps"].append(name)
        if gap < clear_pt - 0.01 or edge < clear_pt - 0.01:
            out["crowded"].append(name)
        if edge < 0:
            out["outside"].append(name)
        if areas and _enters(areas, _box_probes(lo_x, hi_x, lo_y, hi_y))[0]:
            out["inside_area"].append(name)
        if not _owned(_ownership(text, lines_by_color), lo_x, hi_x, lo_y, hi_y)[0]:
            out["misowned"].append(name)
    return out


def problems(audit: dict) -> list[str]:
    """The audit's failures as messages (empty when every text is clear)."""
    found = []
    for key in ("overlaps", "crowded", "outside", "inside_area", "misowned"):
        if audit[key]:
            found.append(f"{key}: {', '.join(audit[key])}")
    return found
