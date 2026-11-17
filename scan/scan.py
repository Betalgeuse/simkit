#!/usr/bin/env python3
"""scan.py <dir>: exit 1 if any file holds a credential, a private path, or is a key, env, database or data file.
Rules: gitleaks-style token prefixes and assignments, private keys, JWTs, home-directory and env paths."""
import os
import re
import sys

RULES = {
    "token-prefix": r"\b(?:lin_(?:api|oauth)_|gh[pousr]_|github_pat_|glpat-|xox[abprs]-|sk-|AKIA|ASIA|AIza)[\w-]{8,}",
    "private-key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "jwt": r"\beyJ[\w-]{10,}\.[\w-]{10,}",
    "assigned-secret": r"(?i)\b[\w.-]*(?:api[_-]?key|token|secret|passw(?:or)?d|app[_-]?key|app[_-]?secret)\b\s*[:=]\s*[\"'][^\"'$<{\s]{8,}[\"']",
    "account-number": r"(?i)\b(?:account|acct|cano)[_ -]?(?:no|num|number|id)?\s*[:=]\s*[\"']?\d{6,}",
    "home-path": r"(?:/Users/|/home/)[\w.-]+/|~/\.|\$HOME/",
    "env-file": r"(?<![\w-])\.(?:env|dev\.vars)(?:\.[\w-]+)?\b",
}
BAD_FILES = re.compile(r"(?i)(\.env(\..*)?|\.pem|\.key|\.p12|\.sqlite3?|\.db|\.npz|\.parquet|\.csv|\.tsv|\.jsonl|id_[rd]sa.*)$")
SKIP = {".git", "__pycache__", ".venv", "out"}


def scan(root):
    hits = []
    me = os.path.abspath(__file__)
    for d, ds, fs in os.walk(root):
        ds[:] = [x for x in ds if x not in SKIP]
        for f in fs:
            p = os.path.join(d, f)
            if BAD_FILES.search(f):
                hits.append(f"{p}: file type")
            if os.path.abspath(p) == me:
                continue
            try:
                text = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            for name, rx in RULES.items():
                for m in re.finditer(rx, text):
                    line = text.count("\n", 0, m.start()) + 1
                    hits.append(f"{p}:{line}: {name}")
    return hits


if __name__ == "__main__":
    hits = scan(sys.argv[1] if len(sys.argv) > 1 else ".")
    print("\n".join(hits) if hits else "scan ok")
    sys.exit(1 if hits else 0)
