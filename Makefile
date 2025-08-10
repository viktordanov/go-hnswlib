# Platform detection (can be overridden)
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Allow override via environment or command line
PLATFORM ?= $(shell echo $(UNAME_S) | tr '[:upper:]' '[:lower:]')
ARCH ?= $(shell echo $(UNAME_M) | sed 's/x86_64/amd64/' | sed 's/aarch64/arm64/')

CXX := clang++

# Build configuration
PLATFORM_DIR := $(PLATFORM)_$(ARCH)
# Build configuration with architecture-specific optimizations
ifeq ($(ARCH),arm64)
    # ARM64 (Apple Silicon) - optimize for ARM but no SIMD (hnswlib doesn't have NEON implementations)
    CXXFLAGS := -O3 -std=c++11 -fPIC -march=native -mtune=native -DNO_MANUAL_VECTORIZATION
else
    # x86_64 - use SSE/AVX SIMD optimizations
    CXXFLAGS := -O3 -std=c++11 -fPIC -march=native -msse4.1 -mavx -mavx2
    # Uncomment for AVX-512 if your target supports it:
    # CXXFLAGS += -mavx512f
endif
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