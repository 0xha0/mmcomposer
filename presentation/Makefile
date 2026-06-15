TECTONIC ?= tectonic
TECTONIC_CACHE ?= /tmp/mmcomposer-tectonic-cache
TECTONIC_FLAGS ?= -Z shell-escape-cwd=$(CURDIR)
LATEXMK ?= latexmk

.PHONY: all latexmk clean

all: main.pdf

main.pdf: main.tex
	XDG_CACHE_HOME=$(TECTONIC_CACHE) $(TECTONIC) $(TECTONIC_FLAGS) main.tex

latexmk: main.tex
	$(LATEXMK) -pdf -shell-escape -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(RM) -r main.pdf main.aux main.fdb_latexmk *.fls main.log main.nav main.out main.snm main.toc main.vrb main.xdv _minted-main
