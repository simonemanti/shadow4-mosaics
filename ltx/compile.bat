@echo off

pdflatex --interaction=nonstopmode main.tex
bibtex main
timeout /t 1 /nobreak >nul


pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

start main.pdf

