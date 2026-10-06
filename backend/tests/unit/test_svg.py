import xml.etree.ElementTree as ET

import pytest

from app import svg

NS = 'xmlns="http://www.w3.org/2000/svg"'


def clean(body: str, attrs: str = "") -> str:
    return svg.sanitise(f"<svg {NS} {attrs}>{body}</svg>")


def test_keeps_plain_shapes_gradients_and_internal_references():
    markup = (f'<svg {NS} xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 256 256">'
              '<defs><linearGradient id="g"><stop offset="0" stop-color="red"/></linearGradient>'
              '<clipPath id="c"><circle r="5"/></clipPath></defs>'
              '<g clip-path="url(#c)"><rect id="r" fill="url(\'#g\')" style="fill: url(#g)"/></g>'
              '<use xlink:href="#r"/><text x="1">A &amp; B</text></svg>')
    out = svg.sanitise(markup)
    root = ET.fromstring(out)
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert [el.tag.rpartition("}")[2] for el in root.iter()] == [
        "svg", "defs", "linearGradient", "stop", "clipPath", "circle", "g", "rect", "use", "text"]
    assert 'href="#r"' in out and 'fill="url(\'#g\')"' in out and 'clip-path="url(#c)"' in out
    assert "A &amp; B" in out
    assert svg.sanitise(out) == out  # stable


def test_unnamespaced_svg_gets_the_namespace():
    assert svg.sanitise('<svg><circle r="1"/></svg>') == f'<svg {NS}><circle r="1" /></svg>'


@pytest.mark.parametrize("body", [
    "<script>alert(1)</script>",
    '<script xlink:href="https://evil.example/x.js" xmlns:xlink="http://www.w3.org/1999/xlink"/>',
    "<style>circle { fill: url(https://evil.example/x) }</style>",
    '<foreignObject><div xmlns="http://www.w3.org/1999/xhtml"><img src="x" onerror="alert(1)"/></div></foreignObject>',
    '<a href="javascript:alert(1)"><circle r="1"/></a>',
    '<image href="https://evil.example/x.png"/>',
    '<animate attributeName="href" to="javascript:alert(1)"/>',
    '<set attributeName="onclick" to="alert(1)"/>',
    '<iframe src="https://evil.example"/>',
    '<x:script xmlns:x="http://www.w3.org/2000/svg">alert(1)</x:script>',
    '<h:script xmlns:h="http://www.w3.org/1999/xhtml">alert(1)</h:script>',
])
def test_drops_dangerous_elements_with_their_content(body):
    out = clean(body + '<circle r="1"/>')
    assert out == f'<svg {NS}><circle r="1" /></svg>'


@pytest.mark.parametrize("attrs", [
    'onload="alert(1)"', 'ONLOAD="alert(1)"', 'onclick="alert(1)"', 'onmouseover="x()"',
    'href="javascript:alert(1)"', 'href=" JavaScript:alert(1)"', 'href="https://evil.example/a.svg#x"',
    'href="data:image/svg+xml;base64,PHN2Zy8+"', 'xlink:href="javascript:alert(1)"',
    'fill="url(https://evil.example/x)"', 'fill="URL( //evil.example/x )"', 'filter="url(a.svg#f)"',
    'style="background: url(javascript:alert(1))"', 'style="fill: u\\72l(https://evil.example)"',
    'fill="url(#ok) url(http://x)"', 'inkscape:label="x" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"',
])
def test_drops_dangerous_attributes(attrs):
    if "xlink:" in attrs:
        attrs += ' xmlns:xlink="http://www.w3.org/1999/xlink"'
    out = clean(f'<use {attrs} x="1"/>', attrs)
    assert out == f'<svg {NS}><use x="1" /></svg>'


@pytest.mark.parametrize("markup,message", [
    ('<!DOCTYPE svg [<!ENTITY x "y">]><svg>&x;</svg>', "DOCTYPE or entities"),
    ('<!doctype svg SYSTEM "file:///etc/passwd"><svg/>', "DOCTYPE or entities"),
    ('<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;">]><svg>&lol2;</svg>',
     "DOCTYPE or entities"),
    ("<svg>&xxe;</svg>", "isn't valid SVG"),  # undeclared entity
    ("<svg><circle></svg>", "isn't valid SVG"),
    ("not svg at all", "isn't valid SVG"),
    ("", "isn't valid SVG"),
    ("<html><body><script>alert(1)</script></body></html>", "root element must be <svg>"),
    ('<x:svg xmlns:x="http://evil.example/ns"/>', "root element must be <svg>"),
    (f"<svg {NS}>" + "<g>" * 2400 + "</g>" * 2400 + "</svg>", "nested too deeply"),
    (f"<svg {NS}><desc>{'x' * 20_000}</desc></svg>", "larger than 20 KB"),
    (f"<svg {NS}><desc>{'é' * 10_000}</desc></svg>", "larger than 20 KB"),  # bytes, not characters
])
def test_rejects(markup, message):
    with pytest.raises(ValueError, match=message):
        svg.sanitise(markup)


def test_comments_and_processing_instructions_are_dropped():
    out = svg.sanitise(f'<?xml version="1.0"?><?xml-stylesheet href="https://evil.example/x.css"?>'
                       f'<svg {NS}><!-- hi --><circle r="1"/></svg>')
    assert out == f'<svg {NS}><circle r="1" /></svg>'


def test_emblem_is_deterministic_per_seed_and_sanitises_unchanged_in_meaning():
    a = svg.emblem("Support")
    assert a == svg.emblem("Support") and a != svg.emblem("support")
    assert ET.fromstring(svg.sanitise(a)).get("viewBox") == "0 0 256 256"
    assert len({svg.emblem(str(i)) for i in range(20)}) == 20
