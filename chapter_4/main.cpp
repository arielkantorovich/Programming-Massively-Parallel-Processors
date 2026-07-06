#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "MatrixMultOperations.cuh"

// ---------------------------------------------------------------------------
// CPU reference matrix multiplication used to validate the GPU kernels.
// M: rm x w, N: w x cn, P: rm x cn  (all row-major)
// ---------------------------------------------------------------------------
static void cpuMatMul(const std::vector<float>& M, const std::vector<float>& N,
                       std::vector<float>& P, int rm, int w, int cn)
{
    for (int r = 0; r < rm; ++r)
    {
        for (int c = 0; c < cn; ++c)
        {
            float acc = 0.0f;
            for (int k = 0; k < w; ++k)
            {
                acc += M[r * w + k] * N[k * cn + c];
            }
            P[r * cn + c] = acc;
        }
    }
}

static void fillRandom(std::vector<float>& v)
{
    for (auto& x : v) x = (float)(rand() % 100) / 10.0f - 5.0f;
}

static bool compare(const std::vector<float>& a, const std::vector<float>& b,
                     float tol, const char* name)
{
    float maxDiff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
    {
        maxDiff = fmaxf(maxDiff, fabsf(a[i] - b[i]));
    }
    bool ok = maxDiff <= tol;
    printf("[%s] max abs diff = %g -> %s\n", name, maxDiff, ok ? "PASS" : "FAIL");
    return ok;
}

int main()
{
    bool allOk = true;
    srand(42);

    // ---- Test 1: NaiveMatMul_kernel - non-square, sizes not multiples of TILE_SIZE ----
    {
        const int M_Rows = 137, M_Cols = 91, N_Cols = 63;
        std::vector<float> h_M(M_Rows * M_Cols), h_N(M_Cols * N_Cols);
        std::vector<float> h_P(M_Rows * N_Cols), h_Ref(M_Rows * N_Cols);

        fillRandom(h_M);
        fillRandom(h_N);
        cpuMatMul(h_M, h_N, h_Ref, M_Rows, M_Cols, N_Cols);

        NaiveMatMulDevice(h_M.data(), h_N.data(), h_P.data(), M_Rows, M_Cols, N_Cols);

        allOk &= compare(h_P, h_Ref, 1e-2f, "NaiveMatMul_kernel (137x91 * 91x63)");
    }

    // ---- Test 2: NaiveTileMatMul_kernel - square, exact multiple of TILE_SIZE ----
    {
        const int n = TILE_SIZE * 5; // kernel assumes n % TILE_SIZE == 0
        std::vector<float> h_M(n * n), h_N(n * n), h_P(n * n), h_Ref(n * n);

        fillRandom(h_M);
        fillRandom(h_N);
        cpuMatMul(h_M, h_N, h_Ref, n, n, n);

        NaiveTileMatMulDevice(h_M.data(), h_N.data(), h_P.data(), n);

        allOk &= compare(h_P, h_Ref, 1e-1f, "NaiveTileMatMul_kernel (80x80)");
    }

    // ---- Test 3: GeneralTileMatMul_kernel - rectangular, NOT multiples of TILE_SIZE ----
    {
        const int rm = 100, w = 70, cn = 130;
        std::vector<float> h_M(rm * w), h_N(w * cn), h_P(rm * cn), h_Ref(rm * cn);

        fillRandom(h_M);
        fillRandom(h_N);
        cpuMatMul(h_M, h_N, h_Ref, rm, w, cn);

        GeneralTileMatMulDevice(h_M.data(), h_N.data(), h_P.data(), rm, w, cn);

        allOk &= compare(h_P, h_Ref, 1e-1f, "GeneralTileMatMul_kernel (100x70 * 70x130)");
    }

    printf("\nOVERALL: %s\n", allOk ? "ALL PASS" : "SOME FAILED");
    return allOk ? 0 : 1;
}
