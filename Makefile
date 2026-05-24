CAIRO_LIB := $(shell pkg-config --variable=libdir cairo)

html: 
	DYLD_FALLBACK_LIBRARY_PATH=$(CAIRO_LIB) uv run mkdocs serve
