# Platform detection (can be overridden)
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Allow override via environment or command line
PLATFORM ?= $(shell echo $(UNAME_S) | tr '[:upper:]' '[:lower:]')
ARCH ?= $(shell echo $(UNAME_M) | sed 's/x86_64/amd64/' | sed 's/aarch64/arm64/')

CXX := clang++

# Build configuration
PLATFORM_DIR := $(PLATFORM)_$(ARCH)
CXXFLAGS := -O3 -std=c++11 -fPIC
INCLUDES := -Ihnswlib -I.
SRC := wrapper/hnsw_wrapper.cpp
LIB := build/$(PLATFORM_DIR)/libhnsw_wrapper.a

.PHONY: all clean

all: $(LIB)

$(LIB): $(SRC)
	@mkdir -p build/$(PLATFORM_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o build/$(PLATFORM_DIR)/hnsw_wrapper.o
	ar rcs $@ build/$(PLATFORM_DIR)/hnsw_wrapper.o

clean:
	rm -rf build