#!/bin/bash
# Compile the Maya blue book (main.tex) with classic BibTeX (iucr.bst) bibliography.
# Runs: pdflatex -> bibtex -> pdflatex -> pdflatex to resolve citations,
# cross-references and the table of contents.
#
# Usage:  cd ltx && ./compile.sh
set -e
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo "Done -> main.pdf"

okular main.pdf
