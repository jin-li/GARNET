PYTHON ?= python
PYTHONPATH := ../..

.PHONY: all figures clean

all: garnet-vs-femffusion.pdf

figures:
	MPLCONFIGDIR=/tmp/garnet-matplotlib PYTHONPATH=$(PYTHONPATH) $(PYTHON) make_figures.py

build/report.pdf: report.tex sections/*.tex figures
	mkdir -p build
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build report.tex
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build report.tex

garnet-vs-femffusion.pdf: build/report.pdf
	cp build/report.pdf $@

clean:
	rm -rf build garnet-vs-femffusion.pdf figures tables
