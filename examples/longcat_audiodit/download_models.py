# SPDX-License-Identifier: Apache-2.0
"""Download ModelScope artifacts and verify published checkpoint hashes."""

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path

from modelscope import snapshot_download
from modelscope.hub.api import HubApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["1B", "3.5B", "tokenizer"], required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    if args.model == "tokenizer":
        model_id = "google/umt5-base"
        expected_sha256 = None
        patterns = ["*.json", "*.model", "README.md"]
    else:
        model_id = f"meituan-longcat/LongCat-AudioDiT-{args.model}"
        expected_sha256 = {
            "1B": "7f41b20933e4466400b8487fd20ca195efa65c5ca7c61f8e9bba6316aa3edcde",
            "3.5B": "708cdbf0bdf71cd485bf10c70a6043b22729730ef55a6b4bc2de2ce552e52fd8",
        }[args.model]
        patterns = ["*.json", "*.safetensors", "LICENSE", "README.md"]
    destination = args.output_dir / model_id.split("/")[-1]
    inventory = HubApi().get_model_files(
        model_id=model_id, revision=args.revision, recursive=True
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "modelscope-inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n"
    )
    snapshot_download(
        model_id,
        revision=args.revision,
        local_dir=str(destination),
        allow_patterns=patterns,
    )
    hashes: dict[str, str] = {}
    for entry in inventory:
        relative_path = entry["Path"]
        if any(fnmatch.fnmatch(relative_path, pattern) for pattern in patterns):
            path = destination / relative_path
            if not path.is_file() or path.stat().st_size != entry["Size"]:
                raise RuntimeError(
                    f"Missing or incorrectly sized artifact: {relative_path}"
                )
            else:
                pass
            with path.open("rb") as handle:
                if handle.read(64).startswith(
                    b"version https://git-lfs.github.com/spec/"
                ):
                    raise RuntimeError(
                        f"Downloaded an LFS pointer instead of {path.name}"
                    )
                else:
                    pass
                handle.seek(0)
                hashes[relative_path] = hashlib.file_digest(
                    handle, "sha256"
                ).hexdigest()
        else:
            pass
    if (
        expected_sha256 is not None
        and hashes.get("model.safetensors") != expected_sha256
    ):
        raise RuntimeError("Checkpoint SHA256 does not match the published inventory")
    else:
        pass
    manifest = {
        "source": "modelscope",
        "model_id": model_id,
        "revision_requested": args.revision,
        "sha256": hashes,
    }
    (destination / "download-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"directory": str(destination), "manifest": manifest}), flush=True)


if __name__ == "__main__":
    main()
else:
    pass
