#include "GEMM.cuh"
#include <vector>
#include <cmath>
#include <random>

// ---------------------------------------------------------------------------
// CPU reference GEMM: P = M * N
//   M : rm x cm   (row-major)
//   N : cm x cn   (row-major)
//   P : rm x cn   (row-major)
// ---------------------------------------------------------------------------
static void cpuGEMM(std::vector<float>& P,
                    const std::vector<float>& M,
                    const std::vector<float>& N,
                    int rm, int cm, int cn)
{
    for (int r = 0; r < rm; ++r)
        for (int c = 0; c < cn; ++c) {
            float acc = 0.0f;
            for (int k = 0; k < cm; ++k)
                acc += M[r * cm + k] * N[c + k * cn];
            P[r * cn + c] = acc;
        }
}

// Compare GPU result to CPU reference with a relative tolerance.
static bool compare(const std::vector<float>& gpu,
                    const std::vector<float>& ref,
                    const char* name)
{
    double maxAbsErr = 0.0;
    int    badIdx    = -1;
    for (size_t i = 0; i < ref.size(); ++i) {
        double diff = std::fabs((double)gpu[i] - (double)ref[i]);
        double tol  = 1e-3 * (1.0 + std::fabs((double)ref[i]));  // relative + absolute floor
        if (diff > tol && diff > maxAbsErr) { maxAbsErr = diff; badIdx = (int)i; }
    }
    if (badIdx == -1) {
        printf("[PASS] %-28s max error within tolerance\n", name);
        return true;
    }
    printf("[FAIL] %-28s idx=%d gpu=%f ref=%f absErr=%g\n",
           name, badIdx, gpu[badIdx], ref[badIdx], maxAbsErr);
    return false;
}

int main()
{
    // Deliberately non-tile-aligned sizes to exercise boundary guards.
    const int rm = 100;   // rows of M / P
    const int cm = 70;    // cols of M / rows of N
    const int cn = 130;   // cols of N / P

    std::vector<float> hM(rm * cm), hN(cm * cn);
    std::vector<float> hRef(rm * cn);
    std::vector<float> hOut(rm * cn);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : hM) x = dist(rng);
    for (auto& x : hN) x = dist(rng);

    cpuGEMM(hRef, hM, hN, rm, cm, cn);

    // Device buffers
    float *dM, *dN, *dP;
    CUDA_CHECK(cudaMalloc(&dM, hM.size()   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dN, hN.size()   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dP, hRef.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dM, hM.data(), hM.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dN, hN.data(), hN.size() * sizeof(float), cudaMemcpyHostToDevice));

    bool ok = true;

    // --- Kernel 1: Naive ---
    {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((cn + TILE_SIZE - 1) / TILE_SIZE,
                  (rm + TILE_SIZE - 1) / TILE_SIZE);
        CUDA_CHECK(cudaMemset(dP, 0, hRef.size() * sizeof(float)));
        NaiveMatrixMultiplication<<<grid, block>>>(dP, dM, dN, rm, cm, cn);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hOut.data(), dP, hRef.size() * sizeof(float), cudaMemcpyDeviceToHost));
        ok &= compare(hOut, hRef, "NaiveMatrixMultiplication");
    }

    // --- Kernel 2: Tiled ---
    {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((cn + TILE_SIZE - 1) / TILE_SIZE,
                  (rm + TILE_SIZE - 1) / TILE_SIZE);
        CUDA_CHECK(cudaMemset(dP, 0, hRef.size() * sizeof(float)));
        TileMatrixMultiplication<<<grid, block>>>(dP, dM, dN, rm, cm, cn);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hOut.data(), dP, hRef.size() * sizeof(float), cudaMemcpyDeviceToHost));
        ok &= compare(hOut, hRef, "TileMatrixMultiplication");
    }

    // --- Kernel 3: Coarsened (each block covers COARSE_FACTOR column-tiles) ---
    {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((cn + TILE_SIZE * COARSE_FACTOR - 1) / (TILE_SIZE * COARSE_FACTOR),
                  (rm + TILE_SIZE - 1) / TILE_SIZE);
        CUDA_CHECK(cudaMemset(dP, 0, hRef.size() * sizeof(float)));
        CoarseMatrixMultiplication<<<grid, block>>>(dP, dM, dN, rm, cm, cn);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hOut.data(), dP, hRef.size() * sizeof(float), cudaMemcpyDeviceToHost));
        ok &= compare(hOut, hRef, "CoarseMatrixMultiplication");
    }

    CUDA_CHECK(cudaFree(dM));
    CUDA_CHECK(cudaFree(dN));
    CUDA_CHECK(cudaFree(dP));

    printf("\n%s\n", ok ? "ALL KERNELS PASSED" : "SOME KERNELS FAILED");
    return ok ? 0 : 1;
}
