# ==========================================================
# chapter_2 - host C++ build
# Builds:  main.cpp + src/*.cpp   ->   bin/main
# ==========================================================

# ----- Automatic Variables ------------------
# - $@ = the target being built
# - $< = the first prerequisite (the .cpp)
# - $^ = all prerequisites (all the .o files)
# --------------------------------------------

# ---- 1. Toolchain (which compiler + which options) ----
CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Iincludes
#            |          |    |             └ where to find vecAdd.hpp
#            |          |    └ turn on warnings (catches bugs early)
#            |          └ optimization level
#            └ C++ standard
# ---- 2. Project layout ----
SRC_DIR  := src
INC_DIR  := includes
BIN_DIR  := bin
# ---- 3. Collect sources, derive object + exe names ----
SRCS   := main.cpp $(wildcard $(SRC_DIR)/*.cpp)   # main.cpp + every .cpp in src/
OBJS   := $(SRCS:%.cpp=$(BIN_DIR)/%.o)            # src/*.cpp -> bin/src/*.o
TARGET := $(BIN_DIR)/main                         # final program
# ---- 4. Targets that are NOT real files ----
.PHONY: all run clean
# Default goal: just running `make` builds the program
all: $(TARGET)
# ---- 5. LINK: stitch all .o files into one executable ----
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@
# ---- 6. COMPILE: one .cpp  ->  one .o ----
# Header listed as prerequisite => editing vecAdd.hpp forces a rebuild.
$(BIN_DIR)/%.o: %.cpp $(INC_DIR)/vecAdd.hpp
	@mkdir -p $(dir $@)             
	$(CXX) $(CXXFLAGS) -c $< -o $@ 
# ---- 7. Convenience helpers ----
run: $(TARGET)
	./$(TARGET)
clean:
	rm -rf $(BIN_DIR)/*.o $(BIN_DIR)/$(SRC_DIR) $(TARGET)