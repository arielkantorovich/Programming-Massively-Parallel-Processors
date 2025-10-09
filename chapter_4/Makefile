# ========= Compilers & Flags =========
NVCC          := nvcc
NVCCFLAGS     := -O2


# ========= Lists (same order!) =========
TARGETS    := MatrixMult Tile
CUDA_FILES := MatMult.cu TileMatMul.cu
# If you prefer auto-picking all .cu files:
# CUDA_FILES := $(wildcard *.cu)
# TARGETS    := color2gray imgBlur      # must have same count as CUDA_FILES

# Sanity check: lengths must match
ifneq ($(words $(TARGETS)),$(words $(CUDA_FILES)))
$(error TARGETS and CUDA_FILES must have the same number of items)
endif

# Pair targets with sources element-wise: "tgt:src" "tgt:src" ...
PAIRS := $(join $(addsuffix :,$(TARGETS)),$(CUDA_FILES))

.PHONY: all clean
all: $(TARGETS)

# Emit one rule per "tgt:src" pair using foreach
define RULE_TEMPLATE
$(1): $(2)
	$$(NVCC) $$(NVCCFLAGS) $(2) -o $(1)
endef
$(foreach P,$(PAIRS), \
  $(eval $(call RULE_TEMPLATE,$(word 1,$(subst :, ,$(P))),$(word 2,$(subst :, ,$(P))))) \
)

clean:
	rm -f $(TARGETS)
