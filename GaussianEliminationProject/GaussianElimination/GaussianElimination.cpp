#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <omp.h>
#include <random>

using namespace std;

// Function to print matrix (for debugging small matrices)
void printMatrix(const vector<vector<double>>& mat, int n, int m) {
    if (n > 10) return; // Don't print large matrices
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cout << fixed << setprecision(2) << mat[i][j] << " ";
        }
        cout << "\n";
    }
    cout << "\n";
}

// Generate random matrix and vector for Ax = b system
void generateSystem(vector<vector<double>>& A, vector<double>& b, int n) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(1.0, 10.0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = dis(gen);
        }
        b[i] = dis(gen);
    }
}

// ====================== SERIAL VERSION ======================

// Serial Gaussian Elimination with Partial Pivoting
void gaussianEliminationSerial(vector<vector<double>>& A, vector<double>& b, int n) {
    // Forward Elimination
    for (int col = 0; col < n - 1; col++) {
        // Find pivot (partial pivoting for stability)
        int pivotRow = col;
        for (int row = col + 1; row < n; row++) {
            if (abs(A[row][col]) > abs(A[pivotRow][col])) {
                pivotRow = row;
            }
        }

        // Swap rows
        swap(A[col], A[pivotRow]);
        swap(b[col], b[pivotRow]);

        // Check for singular matrix
        if (abs(A[col][col]) < 1e-10) {
            cout << "Matrix is singular!\n";
            return;
        }

        // Elimination
        for (int row = col + 1; row < n; row++) {
            double factor = A[row][col] / A[col][col];
            for (int j = col; j < n; j++) {
                A[row][j] -= factor * A[col][j];
            }
            b[row] -= factor * b[col];
        }
    }
}

// Serial Back Substitution
void backSubstitutionSerial(const vector<vector<double>>& A, const vector<double>& b,
    vector<double>& x, int n) {
    x.assign(n, 0.0);

    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }
}

// ====================== PARALLEL VERSION (OpenMP) ======================

// Parallel Gaussian Elimination with Partial Pivoting
void gaussianEliminationParallel(vector<vector<double>>& A, vector<double>& b, int n) {
    // Forward Elimination
    for (int col = 0; col < n - 1; col++) {
        // Find pivot (sequential for simplicity)
        int pivotRow = col;
        for (int row = col + 1; row < n; row++) {
            if (abs(A[row][col]) > abs(A[pivotRow][col])) {
                pivotRow = row;
            }
        }

        // Swap rows
        swap(A[col], A[pivotRow]);
        swap(b[col], b[pivotRow]);

        // Check for singular matrix
        if (abs(A[col][col]) < 1e-10) {
            cout << "Matrix is singular!\n";
            return;
        }

        // Parallel Elimination: distribute row operations across threads
#pragma omp parallel for schedule(dynamic)
        for (int row = col + 1; row < n; row++) {
            double factor = A[row][col] / A[col][col];
            for (int j = col; j < n; j++) {
                A[row][j] -= factor * A[col][j];
            }
            b[row] -= factor * b[col];
        }
    }
}

// Parallel Back Substitution (limited parallelism - sequential dependency)
void backSubstitutionParallel(const vector<vector<double>>& A, const vector<double>& b,
    vector<double>& x, int n) {
    x.assign(n, 0.0);

    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
        for (int j = i + 1; j < n; j++) {
            sum += A[i][j] * x[j];
        }
        x[i] -= sum;
        x[i] /= A[i][i];
    }
}

// ====================== VERIFICATION & PERFORMANCE ANALYSIS ======================

// Verify solution by checking Ax = b
double verifySolution(const vector<vector<double>>& A_original, const vector<double>& b_original,
    const vector<double>& x, int n) {
    double maxError = 0.0;
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A_original[i][j] * x[j];
        }
        double error = abs(sum - b_original[i]);
        maxError = max(maxError, error);
    }
    return maxError;
}

// Main performance testing
int main() {
    vector<int> sizes = { 100, 500, 1000, 2000 };

    cout << "======================================================\n";
    cout << "Gaussian Elimination: Serial vs OpenMP Performance\n";
    cout << "======================================================\n\n";

    for (int n : sizes) {
        cout << "Matrix Size: " << n << " x " << n << "\n";
        cout << "------------------------------------------------------\n";

        // Create matrices
        vector<vector<double>> A_original(n, vector<double>(n));
        vector<double> b_original(n);

        // Generate random system
        generateSystem(A_original, b_original, n);

        // ===== SERIAL VERSION =====
        vector<vector<double>> A_serial = A_original;
        vector<double> b_serial = b_original;
        vector<double> x_serial(n);

        double start = omp_get_wtime();
        gaussianEliminationSerial(A_serial, b_serial, n);
        backSubstitutionSerial(A_serial, b_serial, x_serial, n);
        double serialTime = omp_get_wtime() - start;

        double serialError = verifySolution(A_original, b_original, x_serial, n);

        // ===== PARALLEL VERSION =====
        vector<vector<double>> A_parallel = A_original;
        vector<double> b_parallel = b_original;
        vector<double> x_parallel(n);

        start = omp_get_wtime();
        gaussianEliminationParallel(A_parallel, b_parallel, n);
        backSubstitutionParallel(A_parallel, b_parallel, x_parallel, n);
        double parallelTime = omp_get_wtime() - start;

        double parallelError = verifySolution(A_original, b_original, x_parallel, n);

        // ===== PERFORMANCE METRICS =====
        double speedup = serialTime / parallelTime;
        int numThreads = omp_get_max_threads();
        double efficiency = (speedup / numThreads) * 100.0;

        cout << fixed << setprecision(6);
        cout << "Serial Time:          " << serialTime << " seconds\n";
        cout << "Parallel Time:        " << parallelTime << " seconds\n";
        cout << "Speedup:              " << speedup << "x\n";
        cout << "Number of Threads:    " << numThreads << "\n";
        cout << "Efficiency:           " << efficiency << "%\n";
        cout << "Serial Error:         " << serialError << "\n";
        cout << "Parallel Error:       " << parallelError << "\n";
        cout << "\n";
    }

    return 0;
}