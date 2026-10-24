"""Download the newest snapshot of each rival repo into rivals_live/ (gitignored). Stdlib only.

    python3 -I ftw/fetch_rivals.py [--timeout 5] [--total 20]

Run by run.sh before the bot starts. Contacts only github.com (ref list), api.github.com (fallback) and codeload.github.com. Each repo
gets ~5 s and the whole run at most --total s; anything slower or broken is skipped and RivalHost
falls back to the vendored rivals/<x>. A snapshot whose commit matches the vendored one is not kept
(no point). Never raises; exit status 0.
"""
import io
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from ftw.roster import RIVALS  # noqa: E402
REPOS = [v[1] for v in RIVALS.values()]
UA = {"User-Agent": "ftw-fetch-rivals", "Accept": "application/vnd.github+json"}


def _is_sha(sha):
    return isinstance(sha, str) and len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)


def refs_sha(repo, timeout=5.0):
    """HEAD's commit from git's smart-HTTP ref list (what `git ls-remote` reads; not the API's 60/h limit)."""
    url = "https://github.com/%s.git/info/refs?service=git-upload-pack" % repo
    req = urllib.request.Request(url, headers={"User-Agent": "git/2.40 ftw-fetch-rivals"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read(1 << 20).decode("latin-1")
    for line in body.split("\n"):
        for name in (" HEAD\x00", " HEAD"):
            i = line.find(name)
            if i >= 40 and (line[i:] == " HEAD" or name.endswith("\x00")):
                sha = line[i - 40:i]
                if _is_sha(sha):
                    return sha
    raise ValueError("no HEAD in ref list")


def latest_sha(repo, timeout=5.0):
    try:
        return refs_sha(repo, timeout=timeout)
    except Exception:
        pass
    req = urllib.request.Request("https://api.github.com/repos/%s/commits/HEAD" % repo, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        sha = json.load(r).get("sha")
    if not _is_sha(sha):
        raise ValueError("bad sha from API")
    return sha


def download(repo, dest, ref="HEAD", timeout=5.0, max_bytes=64 << 20):
    """Fetch repo@ref as a tarball and extract it verbatim into dest (replaced). Returns the commit sha
    (from the tarball's pax comment) or ref."""
    url = "https://codeload.github.com/%s/tar.gz/%s" % (repo, ref)
    req = urllib.request.Request(url, headers={"User-Agent": "ftw-fetch-rivals"})
    t_end = time.time() + timeout
    buf = io.BytesIO()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            buf.write(chunk)
            if buf.tell() > max_bytes or time.time() > t_end:
                raise IOError("tarball too big or too slow")
    buf.seek(0)
    tmp = dest + ".part"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    with tarfile.open(fileobj=buf, mode="r:gz") as tf:
        sha = (tf.pax_headers or {}).get("comment") or ref
        for m in tf.getmembers():
            parts = m.name.split("/", 1)
            if len(parts) < 2 or not parts[1]:
                continue
            rel = os.path.normpath(parts[1])
            if rel.startswith("..") or os.path.isabs(rel) or not (m.isfile() or m.isdir()):
                continue            # no links, devices or path tricks
            out = os.path.join(tmp, rel)
            if m.isdir():
                os.makedirs(out, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with tf.extractfile(m) as src, open(out, "wb") as f:
                shutil.copyfileobj(src, f)
            os.chmod(out, 0o644)
    with open(os.path.join(tmp, ".ftw-sha"), "w") as f:
        f.write(sha + "\n")
    shutil.rmtree(dest, ignore_errors=True)
    os.rename(tmp, dest)
    return sha


def snap_name(repo):
    return repo.replace("/", "__", 1)


def read_sha(d):
    try:
        with open(os.path.join(d, ".ftw-sha")) as f:
            return f.read().strip()
    except OSError:
        return None


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--total", type=float, default=20.0)
    a = ap.parse_args(argv)
    end = time.time() + a.total
    live = os.path.join(ROOT, "rivals_live")
    os.makedirs(live, exist_ok=True)
    for repo in REPOS:
        left = end - time.time()
        if left < 1:
            print("fetch_rivals: out of time, skipping %s" % repo, flush=True)
            continue
        t = min(a.timeout, left)
        d = os.path.join(live, snap_name(repo))
        try:
            sha = latest_sha(repo, timeout=t)
            if sha in (read_sha(d), read_sha(os.path.join(ROOT, "rivals", snap_name(repo)))):
                if sha == read_sha(os.path.join(ROOT, "rivals", snap_name(repo))):
                    shutil.rmtree(d, ignore_errors=True)    # vendored copy is current
                print("fetch_rivals: %s up to date (%s)" % (repo, sha[:7]), flush=True)
                continue
            got = download(repo, d, ref=sha, timeout=max(1.0, min(a.timeout, end - time.time())))
            print("fetch_rivals: %s -> rivals_live (%s)" % (repo, got[:7]), flush=True)
        except Exception as e:
            print("fetch_rivals: %s skipped (%s: %s)" % (repo, type(e).__name__, e), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("fetch_rivals: failed (%r)" % (e,))
        sys.exit(0)
