# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/comfy_graph.py - Comfy workflow helpers: UI JSON -> API graph, node lookup, patching.

The user's own workflows (e.g. DepthCrafter, SPEC 4e.1) are stored in UI format. The tool converts them
to the API format without changing any parameter except the ones it is allowed to set (paths, frame range).
UI inputs carry their widget names (`inputs[i].widget.name`), so the conversion needs no /object_info.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

_SEED_CONTROL = {"fixed", "randomize", "increment", "decrement"}


def load_workflow(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _upstream(nodes: Dict[int, dict], links: Dict[int, list], roots: Iterable[int]) -> Set[int]:
    keep: Set[int] = set()
    stack = list(roots)
    while stack:
        nid = stack.pop()
        if nid in keep or nid not in nodes:
            continue
        keep.add(nid)
        for inp in nodes[nid].get("inputs", []) or []:
            lid = inp.get("link")
            if lid is not None and lid in links:
                stack.append(links[lid][1])
    return keep


def ui_to_api(ui: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Comfy UI workflow to an API prompt graph.

    Only nodes that feed an output node (a node without outputs, e.g. a saver) are kept; muted (mode 2)
    and bypassed (mode 4) nodes are dropped. Widget values are mapped in order onto widget inputs.
    """
    if "nodes" not in ui:  # already API format
        return copy.deepcopy(ui)
    nodes = {int(n["id"]): n for n in ui["nodes"] if n.get("mode", 0) not in (2, 4)}
    links = {int(l[0]): l for l in ui.get("links", [])}  # [id, src_node, src_slot, dst_node, dst_slot, type]
    roots = [nid for nid, n in nodes.items() if not n.get("outputs")]
    keep = _upstream(nodes, links, roots)

    api: Dict[str, Any] = {}
    for nid in sorted(keep):
        n = nodes[nid]
        inputs: Dict[str, Any] = {}
        values = list(n.get("widgets_values") or []) if isinstance(n.get("widgets_values"), list) else []
        vi = 0
        for inp in n.get("inputs", []) or []:
            name = inp["name"]
            lid = inp.get("link")
            if lid is not None and lid in links:
                src = links[lid]
                inputs[name] = [str(src[1]), int(src[2])]
                if inp.get("widget"):
                    vi += 1  # widget converted to input still occupies a value slot
                    if name in ("seed", "noise_seed") and vi < len(values) and values[vi] in _SEED_CONTROL:
                        vi += 1
                continue
            if inp.get("widget"):
                if vi < len(values):
                    inputs[name] = values[vi]
                vi += 1
                if name in ("seed", "noise_seed") and vi < len(values) and values[vi] in _SEED_CONTROL:
                    vi += 1
        if isinstance(n.get("widgets_values"), dict):  # e.g. VHS nodes store a dict
            for k, v in n["widgets_values"].items():
                inputs.setdefault(k, v)
        api[str(nid)] = {"class_type": n["type"], "inputs": inputs, "_meta": {"title": n.get("title", n["type"])}}
    return api


def find_nodes(graph: Dict[str, Any], class_type: str) -> List[Tuple[str, Dict[str, Any]]]:
    return [(k, v) for k, v in graph.items() if v.get("class_type") == class_type]


def one_node(graph: Dict[str, Any], class_type: str) -> Tuple[str, Dict[str, Any]]:
    found = find_nodes(graph, class_type)
    if len(found) != 1:
        raise ValueError(f"workflow must contain exactly one {class_type} node (found {len(found)})")
    return found[0]
