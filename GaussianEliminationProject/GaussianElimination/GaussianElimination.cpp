#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <omp.h>
#include <mpi.h>
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

// ====================== MPI VERSION ======================

// MPI Gaussian Elimination with Row Distribution
void gaussianEliminationMPI(vector<vector<double>>& A, vector<double>& b, int n,
    int rank, int size) {
    // Forward Elimination
    for (int col = 0; col < n - 1; col++) {
        // Process that owns the pivot row
        int pivotOwner = col % size;

        int pivotRow = col;
        double maxVal = 0.0;

        // Find local pivot
        if (rank == pivotOwner) {
            maxVal = abs(A[col][col]);
            for (int row = col + 1; row < n; row++) {
                if (row % size == rank && abs(A[row][col]) > maxVal) {
                    maxVal = abs(A[row][col]);
                    pivotRow = row;
                }
            }
        }

        // Broadcast pivot row information
        MPI_Bcast(&pivotRow, 1, MPI_INT, pivotOwner, MPI_COMM_WORLD);

        // Swap rows if necessary (simplified - only owner performs swap)
        int swapOwner = pivotRow % size;
        if (rank == pivotOwner || rank == swapOwner) {
            if (pivotRow != col) {
                swap(A[col], A[pivotRow]);
                swap(b[col], b[pivotRow]);
            }
        }

        // Broadcast pivot row to all processes
        MPI_Bcast(A[col].data(), n, MPI_DOUBLE, pivotOwner, MPI_COMM_WORLD);
        MPI_Bcast(&b[col], 1, MPI_DOUBLE, pivotOwner, MPI_COMM_WORLD);

        // Each process eliminates its own rows
        for (int row = col + 1; row < n; row++) {
            if (row % size == rank) {
                double factor = A[row][col] / A[col][col];
                for (int j = col; j < n; j++) {
                    A[row][j] -= factor * A[col][j];
                }
                b[row] -= factor * b[col];
            }
        }
    }
}

// MPI Back Substitution
void backSubstitutionMPI(const vector<vector<double>>& A, const vector<double>& b,
    vector<double>& x, int n, int rank, int size) {
    x.assign(n, 0.0);

    for (int i = n - 1; i >= 0; i--) {
        int owner = i % size;

        if (rank == owner) {
            x[i] = b[i];
            for (int j = i + 1; j < n; j++) {
                x[i] -= A[i][j] * x[j];
            }
            x[i] /= A[i][i];
        }

        // Broadcast the computed x[i] to all processes
        MPI_Bcast(&x[i], 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
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
int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    vector<int> sizes = { 100, 500, 1000, 2000 };

    if (rank == 0) {
        cout << "======================================================\n";
        cout << "Gaussian Elimination: Serial vs OpenMP vs MPI\n";
        cout << "======================================================\n\n";
    }

    for (int n : sizes) {
        if (rank == 0) {
            cout << "Matrix Size: " << n << " x " << n << "\n";
            cout << "------------------------------------------------------\n";
        }

        // Create matrices
        vector<vector<double>> A_original(n, vector<double>(n));
        vector<double> b_original(n);

        // Generate random system (only on rank 0)
        if (rank == 0) {
            generateSystem(A_original, b_original, n);
        }

        // Broadcast matrix to all processes
        for (int i = 0; i < n; i++) {
            MPI_Bcast(A_original[i].data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        MPI_Bcast(b_original.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // ===== SERIAL VERSION (only rank 0) =====
        double serialTime = 0.0, serialError = 0.0;
        if (rank == 0) {
            vector<vector<double>> A_serial = A_original;
            vector<double> b_serial = b_original;
            vector<double> x_serial(n);

            double start = omp_get_wtime();
            gaussianEliminationSerial(A_serial, b_serial, n);
            backSubstitutionSerial(A_serial, b_serial, x_serial, n);
            serialTime = omp_get_wtime() - start;

            serialError = verifySolution(A_original, b_original, x_serial, n);
        }

        // ===== PARALLEL VERSION (OpenMP - only rank 0) =====
        double parallelTime = 0.0, parallelError = 0.0;
        if (rank == 0) {
            vector<vector<double>> A_parallel = A_original;
            vector<double> b_parallel = b_original;
            vector<double> x_parallel(n);

            double start = omp_get_wtime();
            gaussianEliminationParallel(A_parallel, b_parallel, n);
            backSubstitutionParallel(A_parallel, b_parallel, x_parallel, n);
            parallelTime = omp_get_wtime() - start;

            parallelError = verifySolution(A_original, b_original, x_parallel, n);
        }

        // ===== MPI VERSION (all processes) =====
        vector<vector<double>> A_mpi = A_original;
        vector<double> b_mpi = b_original;
        vector<double> x_mpi(n);

        MPI_Barrier(MPI_COMM_WORLD);
        double mpiStart = MPI_Wtime();

        gaussianEliminationMPI(A_mpi, b_mpi, n, rank, mpi_size);
        backSubstitutionMPI(A_mpi, b_mpi, x_mpi, n, rank, mpi_size);

        MPI_Barrier(MPI_COMM_WORLD);
        double mpiTime = MPI_Wtime() - mpiStart;

        double mpiError = 0.0;
        if (rank == 0) {
            mpiError = verifySolution(A_original, b_original, x_mpi, n);
        }

        // ===== PERFORMANCE METRICS (only rank 0 prints) =====
        if (rank == 0) {
            double speedupOpenMP = serialTime / parallelTime;
            double speedupMPI = serialTime / mpiTime;
            int numThreads = omp_get_max_threads();
            double efficiencyOpenMP = (speedupOpenMP / numThreads) * 100.0;
            double efficiencyMPI = (speedupMPI / mpi_size) * 100.0;

            cout << fixed << setprecision(6);
            cout << "Serial Time:          " << serialTime << " seconds\n";
            cout << "OpenMP Time:          " << parallelTime << " seconds (Threads: " << numThreads << ")\n";
            cout << "MPI Time:             " << mpiTime << " seconds (Processes: " << mpi_size << ")\n";
            cout << "OpenMP Speedup:       " << speedupOpenMP << "x (Efficiency: " << efficiencyOpenMP << "%)\n";
            cout << "MPI Speedup:          " << speedupMPI << "x (Efficiency: " << efficiencyMPI << "%)\n";
            cout << "Serial Error:         " << serialError << "\n";
            cout << "OpenMP Error:         " << parallelError << "\n";
            cout << "MPI Error:            " << mpiError << "\n";
            cout << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}