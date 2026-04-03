import argparse
import zipfile
from pathlib import Path

import browser_cookie3
import requests
import tbdb

TRANSCRIPT_BASE_URL = "https://git.talkbank.org/{}/data"
MEDIA_BASE_URL = "https://media.talkbank.org/"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download TALKBank CHILDES transcripts and their corresponding media "
            "using TBDB metadata and browser cookies."
        )
    )
    parser.add_argument(
        "--browser",
        default="brave",
        help="Browser backend for browser_cookie3 (e.g. brave, chrome, firefox).",
    )
    parser.add_argument(
        "--corpus-name",
        default="childes",
        help="Corpus name to query with TBDB.",
    )
    parser.add_argument(
        "--subset-name",
        dest="subset_names",
        nargs="+",
        default=["Eng-AAE", "Eng-NA", "Eng-UK"],
        help="One or more subset names to download.",
    )
    parser.add_argument(
        "--output-dir",
        default="/Volumes/data",
        help="Root directory for all downloaded files.",
    )
    return parser.parse_args()


def build_session(browser_name: str) -> requests.Session:
    try:
        cookie_loader = getattr(browser_cookie3, browser_name)
    except AttributeError as exc:
        raise ValueError(
            f"Unsupported browser_cookie3 backend: {browser_name}"
        ) from exc

    session = requests.Session()
    session.cookies.update(cookie_loader(domain_name="talkbank.org"))
    return session


def get_transcripts(corpus_name: str, subset_name: str) -> dict:
    spec = {
        "corpusName": corpus_name,
        "corpora": [[corpus_name, subset_name]],
    }
    return tbdb.getTranscripts(spec)


def download_dataset_zip(
    session: requests.Session,
    url: str,
    subset_name: str,
    dataset_name: str,
    transcript_root: Path,
) -> None:
    subset_root = transcript_root / subset_name
    dataset_root = subset_root / dataset_name
    if dataset_root.exists():
        print("SKIP transcript dataset exists:", dataset_root)
        return

    zip_path = subset_root / f"{dataset_name}.zip"
    subset_root.mkdir(parents=True, exist_ok=True)

    print("GET transcript", url)
    response = session.get(url, allow_redirects=True, stream=True)
    ctype = response.headers.get("Content-Type", "")
    print(
        "  status:", response.status_code, "final_url:", response.url, "ctype:", ctype
    )

    if "text/html" in ctype:
        head = next(response.iter_content(chunk_size=4000), b"")
        print("  skip transcript: got html. head:", head[:200])
        return

    if response.status_code not in (200, 206):
        print("  skip transcript:", response.status_code)
        return

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(subset_root)
        zip_path.unlink()
        print("  extracted transcript dataset to", subset_root)
    except zipfile.BadZipFile:
        print("  ERROR: bad transcript zip, keeping", zip_path)


def media_relative_path(
    corpus_name: str, subset_name: str, rel_path: str, media_type: str
):
    if media_type == "audio":
        media_rel = f"{rel_path}.mp3"
    elif media_type == "video":
        media_rel = f"{rel_path}.mp4"
    else:
        return None

    prefix = f"{corpus_name}/{subset_name}/"
    if not media_rel.startswith(prefix):
        return None
    return media_rel.removeprefix(prefix)


def download_media_file(
    session: requests.Session,
    corpus_name: str,
    subset_name: str,
    rel_path: str,
    media_type: str,
    media_root: Path,
) -> None:
    local_rel = media_relative_path(corpus_name, subset_name, rel_path, media_type)
    if local_rel is None:
        return

    out_path = media_root / subset_name / local_rel
    if out_path.exists():
        print("SKIP media exists:", out_path)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{MEDIA_BASE_URL}{rel_path}.{out_path.suffix.lstrip('.')}?f=save"

    print("GET media", url)
    response = session.get(url, stream=True)
    ctype = response.headers.get("Content-Type", "")
    print("  status:", response.status_code, "ctype:", ctype)

    if response.status_code not in (200, 206):
        print("  skip media:", response.status_code)
        return

    if "text/html" in ctype:
        print("  skip media: looks like login page or error")
        return

    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def main():
    args = parse_args()
    transcript_root = Path(args.output_dir) / args.corpus_name / "transcripts"
    media_root = Path(args.output_dir) / args.corpus_name / "media"
    transcript_root.mkdir(parents=True, exist_ok=True)
    media_root.mkdir(parents=True, exist_ok=True)

    session = build_session(args.browser)

    for subset_name in args.subset_names:
        print(f"=== {subset_name} ===")
        transcripts = get_transcripts(args.corpus_name, subset_name)
        col_heads = transcripts["colHeadings"]
        idx_path = col_heads.index("path")
        idx_media = col_heads.index("media")

        dataset_names = set()
        for row in transcripts["data"]:
            rel_path = Path(row[idx_path]).relative_to(
                f"{args.corpus_name}/{subset_name}"
            )
            dataset_names.add(rel_path.parts[0])

        for dataset_name in sorted(dataset_names):
            url = f"{TRANSCRIPT_BASE_URL.format(args.corpus_name)}/{subset_name}/{dataset_name}.zip"
            download_dataset_zip(
                session, url, subset_name, dataset_name, transcript_root
            )

        for row in transcripts["data"]:
            download_media_file(
                session=session,
                corpus_name=args.corpus_name,
                subset_name=subset_name,
                rel_path=row[idx_path],
                media_type=row[idx_media],
                media_root=media_root,
            )


if __name__ == "__main__":
    main()
