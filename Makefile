CXX := clang++
CXXFLAGS := -O3 -std=c++11 -fPIC -Wall -Wextra
INCLUDES := -Ihnswlib -Ihnswlib/hnswlib -I.
SRC := hnsw_wrapper.cpp
OBJ := build/hnsw_wrapper.o
LIB := build/libhnsw_wrapper.a

.PHONY: all clean clean-all clean-objects update-submodule rebuild

all: $(LIB)

build:
	mkdir -p build

$(OBJ): $(SRC) | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(LIB): $(OBJ)
	ar rcs $@ $^

# Remove only intermediate object files (keep static library)
clean-objects:
	rm -f build/*.o
	@echo "✅ Removed object files (kept static library)"

# Remove all build artifacts
clean:
	rm -rf build

# Remove all generated files (build + bindings)
clean-all: clean
	rm -rf bindings/bindings
	@echo "✅ Removed all generated files"

# Update hnswlib submodule to latest version
update-submodule:
	git submodule update --remote hnswlib
	@echo "✅ Updated hnswlib submodule to latest version"
	@echo "🔄 Run './build.sh' to rebuild with updated hnswlib"

# Complete rebuild with updated submodule
rebuild: update-submodule clean
	@echo "🚀 Running complete rebuild..."
	./build.sh