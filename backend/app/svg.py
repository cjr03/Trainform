"""The one SVG sanitiser every stored image (agent images, posts, avatars) passes, and the code-drawn emblem."""

import hashlib
import math
import re
import xml.etree.ElementTree as ET

MAX_BYTES = 20_000
SVG_NS, XLINK_NS = "http://www.w3.org/2000/svg", "http://www.w3.org/1999/xlink"
XML_NS = "http://www.w3.org/XML/1998/namespace"
ELEMENTS = {"svg", "g", "defs", "path", "circle", "ellipse", "rect", "line", "polyline", "polygon", "linearGradient",
            "radialGradient", "stop", "title", "desc", "text", "tspan", "clipPath", "mask", "use"}
INTERNAL_URL = re.compile(r"url\(\s*(['\"]?)#[\w.-]+\1\s*\)", re.I)


def _tag(el) -> str | None:
    """The element's SVG name (with or without the SVG namespace), else None."""
    ns, _, name = el.tag.rpartition("}")
    return name if ns in ("", "{" + SVG_NS) else None


def _attr(name: str, value: str) -> str | None:
    """The name to keep the attribute under, or None to drop it."""
    if name == f"{{{XLINK_NS}}}href":
        name = "href"
    elif name.startswith("{"):
        return name if name.startswith(f"{{{XML_NS}}}") else None  # xml:space etc.; other namespaces' attributes go
    if name.lower().startswith("on") or (name == "href" and not value.strip().startswith("#")):
        return None
    rest = INTERNAL_URL.sub("", value).lower()
    return None if "url(" in rest.replace(" ", "") or "\\" in value else name


def sanitise(markup: str) -> str:
    """`markup` with everything that could run code or load something dropped; ValueError when it can't be."""
    if len(markup.encode()) > MAX_BYTES:
        raise ValueError("The SVG is larger than 20 KB.")
    if re.search(r"<!(DOCTYPE|ENTITY)", markup, re.I):
        raise ValueError("The SVG can't contain a DOCTYPE or entities.")  # XXE and billion laughs
    try:
        root = ET.fromstring(markup)  # comments and processing instructions are dropped
    except ET.ParseError:
        raise ValueError("That isn't valid SVG markup.") from None
    if _tag(root) != "svg":
        raise ValueError("The root element must be <svg>.")
    for el in list(root.iter()):  # iterative: deep nesting can't overflow the stack here
        el.tag = _tag(el)
        for child in list(el):
            if _tag(child) not in ELEMENTS:
                el.remove(child)  # with its subtree: script, style, foreignObject, a, animate, ...
        attrs = {_attr(k, v): v for k, v in el.attrib.items()}
        attrs.pop(None, None)
        el.attrib = attrs
    root.set("xmlns", SVG_NS)
    try:
        return ET.tostring(root, encoding="unicode")
    except RecursionError:
        raise ValueError("The SVG is nested too deeply.") from None


def emblem(seed: str) -> str:
    """A deterministic 256x256 crest seeded by `seed`, for when no model can draw one."""
    h = hashlib.sha256(seed.encode()).digest()
    hue = [h[i] * 360 // 256 for i in range(3)]
    sides = 3 + h[3] % 5
    angles = [2 * math.pi * k / sides - math.pi / 2 for k in range(sides)]
    points = " ".join(f"{128 + 52 * math.cos(a):.1f},{136 + 52 * math.sin(a):.1f}" for a in angles)
    return (f'<svg xmlns="{SVG_NS}" viewBox="0 0 256 256" width="256" height="256">'
            f'<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0" stop-color="hsl({hue[0]},70%,45%)"/><stop offset="1" stop-color="hsl({hue[1]},70%,25%)"/>'
            f'</linearGradient></defs><rect width="256" height="256" rx="40" fill="url(#bg)"/>'
            f'<path d="M128 28 L212 64 V132 C212 184 176 216 128 232 C80 216 44 184 44 132 V64 Z" '
            f'fill="hsl({hue[2]},60%,90%)" stroke="hsl({hue[2]},50%,30%)" stroke-width="8"/>'
            f'<polygon points="{points}" fill="hsl({hue[0]},75%,50%)"/>'
            f'<circle cx="128" cy="136" r="{10 + h[4] % 12}" fill="hsl({hue[1]},80%,95%)"/></svg>')
