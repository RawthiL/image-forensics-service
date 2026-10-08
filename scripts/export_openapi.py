#!/usr/bin/env python3
"""Write docs/openapi.json. Neither Sightengine nor Hive publishes a spec; we do."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ifs.main import create_app  # noqa: E402

out = pathlib.Path(__file__).resolve().parents[1] / "docs" / "openapi.json"
out.write_text(json.dumps(create_app().openapi(), indent=2) + "\n")
print(f"wrote {out}")
