"""Resolving benchmark sources, fetching the remote ones into the output tree.

A benchmark entry in the config may be either a local path or a URL.
"""

from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
from urllib.request import urlretrieve

CHUNK_SIZE = 1 << 16
URL_SCHEMES = frozenset({"http", "https", "ftp", "ftps"})


def is_url(source: str) -> bool:
    """Whether `source` names a URL or a local path."""
    try:
        return urlparse(source).scheme in URL_SCHEMES
    except ValueError:
        return False


def local_name(url: str, label: str) -> str:
    """Cache filename for `url`, prefixed by its benchmark label."""
    filename = PurePosixPath(urlparse(url).path).name
    return f"{label.replace(' ', '_')}_{filename or 'download.vcf.gz'}"


def fetch(source: str, destination_dir: Path, label: str) -> Path:
    """Resolve one benchmark source to a local path.

    A local path is returned unchanged. A URL is downloaded into
    `destination_dir` and the local path returned; an already-present download is
    reused.
    """
    if not is_url(source):
        return Path(source)

    destination = destination_dir / local_name(source, label)
    if destination.exists():
        return destination

    destination_dir.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    print(f"Downloading {label} from {source}")

    try:
        if urlparse(source).scheme in ("ftp", "ftps"):
            urlretrieve(source, partial)
        else:
            import requests

            with requests.get(source, stream=True, timeout=60) as response:
                response.raise_for_status()
                with open(partial, "wb") as handle:
                    handle.writelines(response.iter_content(chunk_size=CHUNK_SIZE))
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    partial.replace(destination)
    print(f"  saved {destination} ({destination.stat().st_size / 1e6:.0f} MB)")
    return destination
