#!/usr/bin/env python3
"""Write docs/openapi.json. Neither Sightengine nor Hive publishes a spec; we do.

The spec is generated from a **pinned canonical configuration**, not from the
operator's `config.yaml`. Two reasons:

* `config.yaml` is gitignored, so it does not exist on a clean checkout or a CI
  runner -- generating from it would simply fail there.
* The app's routes and security scheme depend on configuration (either
  compatibility layer can be switched off, and the API-key header is
  renameable). Generating from a local file would let someone's private
  settings silently drop documented routes from the published spec.

The canonical config enables every endpoint and both compatibility layers,
and uses the `stub` backend, so no weights are fetched and the result
describes the full surface. Inspection's upstreams are left unconfigured:
mounting the route is what shapes the spec, and no service is contacted at
export time.
"""

import json
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CANONICAL_CONFIG = """\
detector:
  backend: stub
  model: stub-v1
auth:
  api_key_header: X-API-Key
endpoints:
  detect:
    enabled: true
    compat:
      sightengine: true
      hive: true
  screen:
    enabled: true
  inspect:
    enabled: true
"""


def main() -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(CANONICAL_CONFIG)
        config_path = fh.name
    os.environ["IFS_CONFIG_FILE"] = config_path
    # Drop any ambient overrides so the spec is identical everywhere.
    for var in [v for v in os.environ if v.startswith("IFS_") and v != "IFS_CONFIG_FILE"]:
        del os.environ[var]

    from ifs.config import reset_settings
    from ifs.main import create_app

    reset_settings()
    out = ROOT / "docs" / "openapi.json"
    out.write_text(json.dumps(create_app().openapi(), indent=2) + "\n")
    os.unlink(config_path)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
