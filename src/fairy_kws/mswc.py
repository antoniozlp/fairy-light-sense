"""Pull individual words out of the Multilingual Spoken Words Corpus (English).

MSWC publishes English audio as one 35 GB `tar.gz` of one-second Opus clips,
`en/clips/<word>/<clip>.opus`, in no particular order. We only want a few
dozen words, so instead of storing the archive we stream it once, keep the
clips we asked for as 16 kHz WAV, and stop as soon as all of them are found.
"""

import gzip
import http.client
import io
import json
import tarfile
import time
import urllib.error
import urllib.request
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm.auto import tqdm

from fairy_kws import config

BASE_URL = "https://mswc.mlcommons-storage.org"
METADATA_URL = f"{BASE_URL}/metadata.json.gz"
AUDIO_URL = f"{BASE_URL}/audio/en.tar.gz"
SPLITS_URL = f"{BASE_URL}/splits/en.tar.gz"

MSWC_DIR = config.DATA_DIR / "mswc"
METADATA_PATH = MSWC_DIR / "metadata.json.gz"
CLIPS_DIR = MSWC_DIR / "clips"
SPLITS_PATH = MSWC_DIR / "en_splits_subset.csv"


def _open(url: str, start: int = 0):
    # The MSWC host rejects urllib's default User-Agent, so send a plain one.
    headers = {"User-Agent": "fairy-kws/0.1", "Range": f"bytes={start}-"}
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)


def download(url: str, path: Path) -> Path:
    """Download `url` to `path` unless it is already there."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with _open(url) as response, open(partial, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=path.name) as bar:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                bar.update(len(chunk))
    partial.rename(path)
    return path


def load_english_index(path: Path = METADATA_PATH) -> dict[str, list[str]]:
    """Return {word: [clip filenames]} for every English word in MSWC."""
    download(METADATA_URL, path)
    with gzip.open(path, "rt") as f:
        return json.load(f)["en"]["filenames"]


def wanted_clips(index: dict[str, list[str]]) -> dict[str, list[str]]:
    """Pick the clips to fetch: the words listed in `config`, capped per word."""
    groups = [
        (config.MSWC_KEYWORD_CANDIDATES, config.MSWC_KEYWORD_CAP),
        (config.MSWC_UNKNOWN_WORDS, config.MSWC_UNKNOWN_CAP),
    ]
    return {
        word: sorted(index[word])[:cap] for words, cap in groups for word in words if word in index
    }


class _ResumingStream(io.RawIOBase):
    """Read a URL sequentially, reconnecting from the current byte if the link drops."""

    def __init__(self, url: str, max_retries: int = 20):
        self.url = url
        self.max_retries = max_retries
        self.position = 0
        self.response = None
        self._connect()
        self.length = self.position + int(self.response.headers["Content-Length"])

    def _connect(self):
        self.response = _open(self.url, self.position)

    def readable(self):
        return True

    def readinto(self, buffer):
        for attempt in range(self.max_retries):
            try:
                if self.response is None:
                    self._connect()
                n = self.response.readinto(buffer)
                if n == 0 and self.position < self.length:
                    raise ConnectionError("stream ended early")
                self.position += n
                return n
            except (OSError, http.client.HTTPException, urllib.error.URLError):
                self.response = None
                time.sleep(min(60, 2**attempt))
        raise ConnectionError(f"gave up on {self.url} at byte {self.position}")


def opus_to_wav(data: bytes, path: Path) -> None:
    """Decode one clip and save it as 16 kHz mono int16 WAV, exactly one second long."""
    audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != config.SAMPLE_RATE:
        divisor = gcd(rate, config.SAMPLE_RATE)
        audio = resample_poly(audio, config.SAMPLE_RATE // divisor, rate // divisor)
    audio = audio[: config.CLIP_SAMPLES]
    audio = np.pad(audio, (0, config.CLIP_SAMPLES - len(audio)))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -1.0, 1.0), config.SAMPLE_RATE, subtype="PCM_16")


def extract_words(
    wanted: dict[str, list[str]],
    out_dir: Path = CLIPS_DIR,
    url: str = AUDIO_URL,
) -> dict[str, int]:
    """Stream the English audio archive and save the clips listed in `wanted`.

    `wanted` maps each word to the Opus filenames to keep (from `load_english_index`).
    Clips already on disk are skipped, and nothing is downloaded when all are present.
    Returns the number of clips on disk per word.
    """
    # Clips a previous complete pass could not get (missing from the archive or
    # undecodable) are remembered, so they do not trigger another 35 GB pass.
    unavailable_path = out_dir / "unavailable.txt"
    unavailable = set(unavailable_path.read_text().split()) if unavailable_path.exists() else set()
    remaining = {
        (word, name)
        for word, names in wanted.items()
        for name in names
        if f"{word}/{name}" not in unavailable
        and not (out_dir / word / name).with_suffix(".wav").exists()
    }
    if remaining:
        failed = set()
        stream = _ResumingStream(url)
        with (
            tqdm(total=stream.length, unit="B", unit_scale=True, mininterval=5) as bar,
            tarfile.open(fileobj=io.BufferedReader(stream, 1 << 20), mode="r|gz") as tar,
        ):
            for i, member in enumerate(tar):
                key = tuple(member.name.split("/")[-2:])
                if key in remaining:
                    try:
                        data = tar.extractfile(member).read()
                        opus_to_wav(data, (out_dir / key[0] / key[1]).with_suffix(".wav"))
                    except (sf.LibsndfileError, ValueError):
                        failed.add(key)
                    remaining.discard(key)
                    if not remaining:
                        break
                if i % 5000 == 0:
                    tar.members = []  # tarfile remembers every entry; drop them to save memory
                    bar.update(stream.position - bar.n)
                    bar.set_postfix(clips_left=len(remaining))
        # Reaching this line means the pass was complete (an interrupted pass raises).
        if failed or remaining:
            print(f"{len(failed)} clips undecodable, {len(remaining)} not in the archive")
            unavailable |= {"/".join(key) for key in failed | remaining}
            out_dir.mkdir(parents=True, exist_ok=True)
            unavailable_path.write_text("\n".join(sorted(unavailable)) + "\n")
    return {word: len(list((out_dir / word).glob("*.wav"))) for word in wanted}


def extract_splits(words: set[str], path: Path = SPLITS_PATH, url: str = SPLITS_URL) -> Path:
    """Save the rows of MSWC's official split table for `words` as a small CSV.

    The table is fetched again when `words` has grown since the CSV was written.

    Columns: SET (TRAIN/DEV/TEST), LINK (<word>/<clip>.opus), WORD, VALID, SPEAKER, GENDER.
    The official split keeps each speaker in a single set.
    """
    if path.exists():
        with open(path) as table:
            saved = {line.split(",", 3)[2] for line in table}
        if words <= saved:
            return path
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = _ResumingStream(url)
    with (
        tqdm(total=stream.length, unit="B", unit_scale=True, mininterval=5) as bar,
        tarfile.open(fileobj=io.BufferedReader(stream, 1 << 20), mode="r|gz") as tar,
    ):
        for member in tar:
            if not member.name.endswith("en_splits.csv"):
                continue
            table = tar.extractfile(member)
            kept, rest, header = [], b"", None
            while chunk := table.read(1 << 24):
                *lines, rest = (rest + chunk).split(b"\n")
                if header is None:
                    header, *lines = lines
                kept += [line for line in lines if line.split(b",", 3)[2].decode() in words]
                bar.update(stream.position - bar.n)
            if rest and rest.split(b",", 3)[2].decode() in words:
                kept.append(rest)
            path.write_bytes(b"\n".join([header, *kept]) + b"\n")
            return path
    raise FileNotFoundError("en_splits.csv not found in the splits archive")


def fetch_all() -> dict[str, int]:
    """Fetch the split table and every clip listed in `config`. Safe to re-run."""
    wanted = wanted_clips(load_english_index())
    extract_splits(set(wanted))
    return extract_words(wanted)


if __name__ == "__main__":
    print(fetch_all())
