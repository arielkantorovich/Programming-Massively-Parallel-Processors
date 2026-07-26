NVCC      := nvcc
NVCCFLAGS := -O2 -std=c++14

# Default build: the thread-coarsening stencil program.
# (shmem_stencil.cu has no main(), so it is not linked into an executable.)
TARGETS := 3D_thread_coearsing basic_stencil

.PHONY: all clean

all: $(TARGETS)

3D_thread_coearsing: 3D_thread_coearsing.cu param_c.hpp
	$(NVCC) $(NVCCFLAGS) $< -o $@

basic_stencil: basic_stencil.cu param_c.hpp
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
