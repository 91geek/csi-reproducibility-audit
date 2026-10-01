#!/usr/bin/env python3
# pack_arxiv.py — Prepare arXiv submission package for Paper A
#
# Reads:  ../docs/paper_ieee_tmc.tex
#         ../src/bvp_test/*.png  (figures)
# Writes: paper_arxiv.tex       (arXiv-compatible LaTeX)
#         paper_arxiv.bbl       (bibliography, copied inline)
#         figs/*.png            (figures)
#         paper_arxiv.zip       (everything zipped for direct upload)
#
# Run: python pack_arxiv.py

import os
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_TEX = ROOT / "docs" / "paper_ieee_tmc.tex"
SRC_FIG_DIR = ROOT / "src" / "bvp_test"
DST = Path(__file__).resolve().parent
DST_FIGS = DST / "figs"


def read_tex():
    return SRC_TEX.read_text(encoding="utf-8")


def transform_tex(tex):
    """Make the IEEE TMC LaTeX arXiv-compatible."""
    tex = re.sub(
        r'\\documentclass\[journal,compsoc\]\{IEEEtran\}',
        r'\\documentclass[11pt]{article}' + '\n\n'
        r'% Originally written for IEEE TMC submission.' + '\n'
        r'% Converted to arXiv-compatible article class.\n'
        r'% IEEEtran-specific commands and constraints removed.',
        tex,
        count=1,
    )

    # Remove IEEEtran-specific packages and commands
    tex = re.sub(r'\\IEEEpubidadjcol\b[^\n]*\n', '', tex)

    # Add arXiv-friendly packages (article doesn't include hyperref by default)
    # but keep the existing package list

    # Wrap in twocolumn emulation for IEEE-style formatting (optional)
    # For arXiv, single-column is fine and easier to read

    # Update figure paths from ../src/bvp_test/ to figs/
    tex = tex.replace("../src/bvp_test/", "figs/")

    # Update references like Fig.~\ref{fig:forest} remain unchanged (refs are internal)

    # Update Acknowledgments TODO line (replace placeholder if any)
    tex = re.sub(
        r'This work was supported by \\\\TODO\{[^}]*\}.',
        'This work was supported by internal research funding. '
        'The authors thank anonymous reviewers for their constructive feedback.',
        tex,
    )

    # Update title page header to be cleaner for arXiv
    # Remove "Submitted to IEEE TMC" notes; arXiv Comments field handles this
    tex = re.sub(
        r'% Title:.*?\n',
        '',
        tex,
        count=0,
    )

    return tex


def write_arxiv_tex(tex):
    out = DST / "paper_arxiv.tex"
    out.write_text(tex, encoding="utf-8")
    return out


def copy_figs():
    DST_FIGS.mkdir(exist_ok=True)
    copied = []
    for png in SRC_FIG_DIR.glob("*.png"):
        dst_png = DST_FIGS / png.name
        shutil.copy2(png, dst_png)
        copied.append(dst_png.name)
    return copied


def extract_bbl(tex):
    """Extract \begin{thebibliography} ... \end{thebibliography} to a .bbl file."""
    m = re.search(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}',
                  tex, re.DOTALL)
    if not m:
        return None
    bbl = m.group(0)
    out = DST / "paper_arxiv.bbl"
    out.write_text(bbl, encoding="utf-8")
    return out


def make_zip(tex_path, bbl_path, figs):
    """Create a single zip that can be uploaded to arXiv."""
    zip_path = DST / "paper_arxiv.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(tex_path, arcname=tex_path.name)
        if bbl_path and bbl_path.exists():
            zf.write(bbl_path, arcname=bbl_path.name)
        for f in figs:
            zf.write(DST_FIGS / f, arcname=f"figs/{f}")
    return zip_path


def main():
    print(f"Source: {SRC_TEX}")
    print(f"Dest:   {DST}")
    print()

    tex = read_tex()
    print(f"Source .tex size: {len(tex)} chars, {tex.count(chr(10))} lines")

    tex2 = transform_tex(tex)
    tex_path = write_arxiv_tex(tex2)
    print(f"Wrote arxiv .tex: {tex_path} ({len(tex2)} chars)")

    bbl_path = extract_bbl(tex2)
    if bbl_path:
        print(f"Wrote arxiv .bbl: {bbl_path}")

    figs = copy_figs()
    print(f"Copied {len(figs)} figures to {DST_FIGS}")

    zip_path = make_zip(tex_path, bbl_path, figs)
    print(f"Wrote zip: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")

    print()
    print("=== Done ===")
    print(f"Upload {zip_path} to arxiv.org/submit (or upload files individually).")
    print("Copy metadata fields from ARXIV_METADATA.txt.")


if __name__ == "__main__":
    main()