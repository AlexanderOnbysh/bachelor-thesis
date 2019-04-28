all: build clean

build:
	latexmk -xelatex -synctex=1 -jobname=bachelor-thesis thesis.tex

run:
	xreader master-thesis.pdf &

clean:
	rm *.aux \
	*.fdb_latexmk \
	*.fls \
	*.lof \
	*.lot \
	*.log \
	*.out \
	*.synctex.gz \
	*.toc \
	*.dvi \
	*.bbl \
	*.blg