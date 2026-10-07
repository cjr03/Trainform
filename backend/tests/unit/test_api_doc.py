import re
from pathlib import Path

from app.main import app

API_MD = Path(__file__).parents[3] / "API.md"


def test_api_md_documents_every_route():
    documented = set(re.findall(r"^### (GET|POST|PUT|PATCH|DELETE) (/api/\S+)", API_MD.read_text(), re.M))
    routes = {(method.upper(), path) for path, ops in app.openapi()["paths"].items() for method in ops}
    assert documented == routes
