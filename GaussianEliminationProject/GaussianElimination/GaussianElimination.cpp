#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <omp.h>
#include <mpi.h>
using namespace std;

// Generate random matrix A and vector b
void generateSystem(vector<vector<double>>& A, vector<double>& b, int n) {
    srand(time(0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (rand() % 100);
        }
        b[i] = (rand() % 100) + 1.0;
    }
}

// ==================== SERIAL IMPLEMENTATION ====================

void serialGaussianElimination(vector<vector<double>>& A, vector<double>& b, int n) {
    // Forward elimination
    for (int col = 0; col < n - 1; col++) {
        // Find the largest element in current column (pivoting)
        int pivotRow = col;
        for (int row = col + 1; row < n; row++) {
            if (abs(A[row][col]) > abs(A[pivotRow][col])) {
                pivotRow = row;
            }
        }

        // Swap rows
        swap(A[col], A[pivotRow]);
        swap(b[col], b[pivotRow]);

        // Check if matrix is singular
        if (abs(A[col][col]) < 0.0000001) {
            cout << "Matrix is singular!" << endl;
            return;
        }

        // Eliminate column entries below pivot
        for (int row = col + 1; row < n; row++) {
            double factor = A[row][col] / A[col][col];
            for (int j = col; j < n; j++) {
                A[row][j] = A[row][j] - factor * A[col][j];
            }
            b[row] = b[row] - factor * b[col];
        }
    }
}

void serialBackSubstitution(vector<vector<double>>& A, vector<double>& b,
    vector<double>& x, int n) {
    // Solve from bottom to top
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) {
            x[i] = x[i] - A[i][j] * x[j];
        }
        x[i] = x[i] / A[i][i];
    }
}

// ==================== OPENMP IMPLEMENTATION ====================

void openmpGaussianElimination(vector<vector<double>>& A, vector<double>& b, int n) {
    // Forward elimination
    for (int col = 0; col < n - 1; col++) {
        // Find pivot (sequential - simpler to understand)
        int pivotRow = col;
        for (int row = col + 1; row < n; row++) {
            if (abs(A[row][col]) > abs(A[pivotRow][col])) {
                pivotRow = row;
            }
        }

        // Swap rows
        swap(A[col], A[pivotRow]);
        swap(b[col], b[pivotRow]);

        if (abs(A[col][col]) < 0.0000001) {
            cout << "Matrix is singular!" << endl;
            return;
        }

        // Parallel elimination using OpenMP
#pragma omp parallel for
        for (int row = col + 1; row < n; row++) {
            double factor = A[row][col] / A[col][col];
            for (int j = col; j < n; j++) {
                A[row][j] = A[row][j] - factor * A[col][j];
            }
            b[row] = b[row] - factor * b[col];
        }
    }
}

void openmpBackSubstitution(vector<vector<double>>& A, vector<double>& b,
    vector<double>& x, int n) {
    // Back substitution (limited parallelism due to dependencies)
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        double sum = 0.0;

        // Parallel reduction for sum
#pragma omp parallel for reduction(+:sum)
        for (int j = i + 1; j < n; j++) {
            sum += A[i][j] * x[j];
        }

        x[i] = (x[i] - sum) / A[i][i];
    }
}

// ==================== MPI IMPLEMENTATION ====================

void mpiGaussianElimination(vector<vector<double>>& A, vector<double>& b, int n,
    int rank, int numProcs) {
    // Forward elimination
    for (int col = 0; col < n - 1; col++) {
        int pivotOwner = col % numProcs;

        // Find pivot row
        int pivotRow = col;
        double maxVal = 0.0;

        if (rank == pivotOwner) {
            maxVal = abs(A[col][col]);
            for (int row = col + 1; row < n; row++) {
                if (row % numProcs == rank && abs(A[row][col]) > maxVal) {
                    maxVal = abs(A[row][col]);
                    pivotRow = row;
                }
            }
        }

        // Broadcast pivot row number
        MPI_Bcast(&pivotRow, 1, MPI_INT, pivotOwner, MPI_COMM_WORLD);

        // Swap rows (simplified - only owner swaps)
        int swapOwner = pivotRow % numProcs;
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
            if (row % numProcs == rank) {
                double factor = A[row][col] / A[col][col];
                for (int j = col; j < n; j++) {
                    A[row][j] = A[row][j] - factor * A[col][j];
                }
                b[row] = b[row] - factor * b[col];
            }
        }
    }
}

void mpiBackSubstitution(vector<vector<double>>& A, vector<double>& b,
    vector<double>& x, int n, int rank, int numProcs) {
    // Back substitution
    for (int i = n - 1; i >= 0; i--) {
        int owner = i % numProcs;

        if (rank == owner) {
            x[i] = b[i];
            for (int j = i + 1; j < n; j++) {
                x[i] = x[i] - A[i][j] * x[j];
            }
            x[i] = x[i] / A[i][i];
        }

        // Broadcast computed value to all processes
        MPI_Bcast(&x[i], 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    }
}

// ==================== VERIFICATION ====================

double checkError(vector<vector<double>>& A_orig, vector<double>& b_orig,
    vector<double>& x, int n) {
    double maxError = 0.0;
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A_orig[i][j] * x[j];
        }
        double error = abs(sum - b_orig[i]);
        if (error > maxError) {
            maxError = error;
        }
    }
    return maxError;
}

// ==================== MAIN PROGRAM ====================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    vector<int> sizes = { 100, 500, 1000, 2000 };

    if (rank == 0) {
        cout << "========================================" << endl;
        cout << "Gaussian Elimination Performance Test" << endl;
        cout << "========================================" << endl << endl;
    }

    for (int n : sizes) {
        if (rank == 0) {
            cout << "Matrix Size: " << n << " x " << n << endl;
            cout << "----------------------------------------" << endl;
        }

        // Create matrices
        vector<vector<double>> A_orig(n, vector<double>(n));
        vector<double> b_orig(n);

        // Generate system (rank 0)
        if (rank == 0) {
            generateSystem(A_orig, b_orig, n);
        }

        // Broadcast to all processes
        for (int i = 0; i < n; i++) {
            MPI_Bcast(A_orig[i].data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        MPI_Bcast(b_orig.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // SERIAL VERSION
        double serialTime = 0.0;
        double serialError = 0.0;
        if (rank == 0) {
            vector<vector<double>> A_serial = A_orig;
            vector<double> b_serial = b_orig;
            vector<double> x_serial(n);

            double start = omp_get_wtime();
            serialGaussianElimination(A_serial, b_serial, n);
            serialBackSubstitution(A_serial, b_serial, x_serial, n);
            serialTime = omp_get_wtime() - start;

            serialError = checkError(A_orig, b_orig, x_serial, n);
        }

        // OPENMP VERSION
        double openmpTime = 0.0;
        double openmpError = 0.0;
        if (rank == 0) {
            vector<vector<double>> A_openmp = A_orig;
            vector<double> b_openmp = b_orig;
            vector<double> x_openmp(n);

            double start = omp_get_wtime();
            openmpGaussianElimination(A_openmp, b_openmp, n);
            openmpBackSubstitution(A_openmp, b_openmp, x_openmp, n);
            openmpTime = omp_get_wtime() - start;

            openmpError = checkError(A_orig, b_orig, x_openmp, n);
        }

        // MPI VERSION
        vector<vector<double>> A_mpi = A_orig;
        vector<double> b_mpi = b_orig;
        vector<double> x_mpi(n);

        MPI_Barrier(MPI_COMM_WORLD);
        double mpiStart = MPI_Wtime();

        mpiGaussianElimination(A_mpi, b_mpi, n, rank, numProcs);
        mpiBackSubstitution(A_mpi, b_mpi, x_mpi, n, rank, numProcs);

        MPI_Barrier(MPI_COMM_WORLD);
        double mpiTime = MPI_Wtime() - mpiStart;

        double mpiError = 0.0;
        if (rank == 0) {
            mpiError = checkError(A_orig, b_orig, x_mpi, n);
        }

        // Print results (rank 0 only)
        if (rank == 0) {
            int numThreads = omp_get_max_threads();
            double speedupOpenMP = serialTime / openmpTime;
            double speedupMPI = serialTime / mpiTime;
            double efficiencyOpenMP = (speedupOpenMP / numThreads) * 100.0;
            double efficiencyMPI = (speedupMPI / numProcs) * 100.0;

            cout << "Serial Time:      " << serialTime << " seconds" << endl;
            cout << "OpenMP Time:      " << openmpTime << " seconds (Threads: "
                << numThreads << ")" << endl;
            cout << "MPI Time:         " << mpiTime << " seconds (Processes: "
                << numProcs << ")" << endl;
            cout << "OpenMP Speedup:   " << speedupOpenMP << "x (Efficiency: "
                << efficiencyOpenMP << "%)" << endl;
            cout << "MPI Speedup:      " << speedupMPI << "x (Efficiency: "
                << efficiencyMPI << "%)" << endl;
            cout << "Serial Error:     " << serialError << endl;
            cout << "OpenMP Error:     " << openmpError << endl;
            cout << "MPI Error:        " << mpiError << endl;
            cout << endl;
        }
    }

    MPI_Finalize();
    return 0;
}