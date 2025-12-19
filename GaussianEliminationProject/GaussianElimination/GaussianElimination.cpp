
#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <omp.h>
#include <mpi.h>
#include <iomanip>
using namespace std;

struct ExperimentResult {
    int matrixSize;
    double serialTime;
    double openmpTime;
    double mpiTime;
    double openmpSpeedup;
    double mpiSpeedup;
    double openmpEfficiency;
    double mpiEfficiency;
};

// Generate random matrix and vector for the linear system
void generateSystem(vector<vector<double>>& coefficientMatrix, vector<double>& constantsVector, int matrixSize) {
    srand(time(0));

    for (int row = 0; row < matrixSize; row++) {
        for (int col = 0; col < matrixSize; col++) {
            coefficientMatrix[row][col] = (rand() % 100);
        }
        constantsVector[row] = (rand() % 100) + 1.0;
    }
}

// ==================== SERIAL IMPLEMENTATION ====================

void serialGaussianElimination(vector<vector<double>>& coefficientMatrix, vector<double>& constantsVector, int matrixSize) {
    // Forward elimination
    for (int col = 0; col < matrixSize - 1; col++) {
        // Find the largest element in current column (pivoting)
        int pivotRowIndex = col;
        for (int row = col + 1; row < matrixSize; row++) {
            if (abs(coefficientMatrix[row][col]) > abs(coefficientMatrix[pivotRowIndex][col])) {
                pivotRowIndex = row;
            }
        }

        // Swap rows
        swap(coefficientMatrix[col], coefficientMatrix[pivotRowIndex]);
        swap(constantsVector[col], constantsVector[pivotRowIndex]);

        // Check if matrix is singular
        if (abs(coefficientMatrix[col][col]) < 0.0000001) {
            cout << "Matrix is singular!" << endl;
            return;
        }

        // Eliminate column entries below pivot
        for (int row = col + 1; row < matrixSize; row++) {
            double eliminationFactor = coefficientMatrix[row][col] / coefficientMatrix[col][col];
            for (int j = col; j < matrixSize; j++) {
                coefficientMatrix[row][j] = coefficientMatrix[row][j] - eliminationFactor * coefficientMatrix[col][j];
            }
            constantsVector[row] = constantsVector[row] - eliminationFactor * constantsVector[col];
        }
    }
}

void serialBackSubstitution(vector<vector<double>>& coefficientMatrix, vector<double>& constantsVector,
    vector<double>& solutionVector, int matrixSize) {
    // Solve from bottom to top
    for (int row = matrixSize - 1; row >= 0; row--) {
        solutionVector[row] = constantsVector[row];
        for (int col = row + 1; col < matrixSize; col++) {
            solutionVector[row] = solutionVector[row] - coefficientMatrix[row][col] * solutionVector[col];
        }
        solutionVector[row] = solutionVector[row] / coefficientMatrix[row][row];
    }
}

// ==================== OPENMP IMPLEMENTATION ====================

void openmpGaussianElimination(vector<vector<double>>& coefficientMatrix, vector<double>& constantsVector, int matrixSize) {
    // Forward elimination
    for (int col = 0; col < matrixSize - 1; col++) {
        // Find pivot (sequential - simpler to understand)
        int pivotRowIndex = col;
        for (int row = col + 1; row < matrixSize; row++) {
            if (abs(coefficientMatrix[row][col]) > abs(coefficientMatrix[pivotRowIndex][col])) {
                pivotRowIndex = row;
            }
        }

        // Swap rows
        swap(coefficientMatrix[col], coefficientMatrix[pivotRowIndex]);
        swap(constantsVector[col], constantsVector[pivotRowIndex]);

        if (abs(coefficientMatrix[col][col]) < 0.0000001) {
            cout << "Matrix is singular!" << endl;
            return;
        }

        // Parallel elimination using OpenMP
#pragma omp parallel for
        for (int row = col + 1; row < matrixSize; row++) {
            double eliminationFactor = coefficientMatrix[row][col] / coefficientMatrix[col][col];
            for (int j = col; j < matrixSize; j++) {
                coefficientMatrix[row][j] = coefficientMatrix[row][j] - eliminationFactor * coefficientMatrix[col][j];
            }
            constantsVector[row] = constantsVector[row] - eliminationFactor * constantsVector[col];
        }
    }
}

void openmpBackSubstitution(vector<vector<double>>& coefficientMatrix, vector<double>& constantsVector,
    vector<double>& solutionVector, int matrixSize) {
    // Back substitution (limited parallelism due to dependencies)
    for (int i = matrixSize - 1; i >= 0; i--) {
        solutionVector[i] = constantsVector[i];
        double rowSum = 0.0;

        // Parallel reduction for rowSum
#pragma omp parallel for reduction(+:rowSum)
        for (int j = i + 1; j < matrixSize; j++) {
            rowSum += coefficientMatrix[i][j] * solutionVector[j];
        }

        solutionVector[i] = (solutionVector[i] - rowSum) / coefficientMatrix[i][i];
    }
}

// ==================== MPI IMPLEMENTATION ====================

void mpiGaussianElimination(vector<vector<double>>& coefficientMatrix, vector<double>& constantsVector, int matrixSize,
    int processRank, int totalProcesses) {
    // Forward elimination
    for (int col = 0; col < matrixSize - 1; col++) {
        int pivotOwnerRank = col % totalProcesses;

        // Find pivot row
        int pivotRowIndex = col;
        double maxAbsoluteValue = 0.0;

        if (processRank == pivotOwnerRank) {
            maxAbsoluteValue = abs(coefficientMatrix[col][col]);
            for (int row = col + 1; row < matrixSize; row++) {
                if (row % totalProcesses == processRank && abs(coefficientMatrix[row][col]) > maxAbsoluteValue) {
                    maxAbsoluteValue = abs(coefficientMatrix[row][col]);
                    pivotRowIndex = row;
                }
            }
        }

        // Broadcast pivot row number
        MPI_Bcast(&pivotRowIndex, 1, MPI_INT, pivotOwnerRank, MPI_COMM_WORLD);

        // Swap rows (simplified - only owner swaps)
        int swapOwnerRank = pivotRowIndex % totalProcesses;
        if (processRank == pivotOwnerRank || processRank == swapOwnerRank) {
            if (pivotRowIndex != col) {
                swap(coefficientMatrix[col], coefficientMatrix[pivotRowIndex]);
                swap(constantsVector[col], constantsVector[pivotRowIndex]);
            }
        }

        // Broadcast pivot row to all processes
        MPI_Bcast(coefficientMatrix[col].data(), matrixSize, MPI_DOUBLE, pivotOwnerRank, MPI_COMM_WORLD);
        MPI_Bcast(&constantsVector[col], 1, MPI_DOUBLE, pivotOwnerRank, MPI_COMM_WORLD);

        // Each process eliminates its own rows
        for (int row = col + 1; row < matrixSize; row++) {
            if (row % totalProcesses == processRank) {
                double eliminationFactor = coefficientMatrix[row][col] / coefficientMatrix[col][col];
                for (int j = col; j < matrixSize; j++) {
                    coefficientMatrix[row][j] = coefficientMatrix[row][j] - eliminationFactor * coefficientMatrix[col][j];
                }
                constantsVector[row] = constantsVector[row] - eliminationFactor * constantsVector[col];
            }
        }
    }
}

void mpiBackSubstitution(vector<vector<double>>& coefficientMatrix, vector<double>& constantsVector,
    vector<double>& solutionVector, int matrixSize, int processRank, int totalProcesses) {
    // Back substitution
    for (int i = matrixSize - 1; i >= 0; i--) {
        int ownerRank = i % totalProcesses;

        if (processRank == ownerRank) {
            solutionVector[i] = constantsVector[i];
            for (int j = i + 1; j < matrixSize; j++) {
                solutionVector[i] = solutionVector[i] - coefficientMatrix[i][j] * solutionVector[j];
            }
            solutionVector[i] = solutionVector[i] / coefficientMatrix[i][i];
        }

        // Broadcast computed value to all processes
        MPI_Bcast(&solutionVector[i], 1, MPI_DOUBLE, ownerRank, MPI_COMM_WORLD);
    }
}

// ==================== VERIFICATION ====================

double checkError(vector<vector<double>>& originalMatrix, vector<double>& originalConstants,
    vector<double>& solutionVector, int matrixSize) {
    double maxError = 0.0;
    for (int row = 0; row < matrixSize; row++) {
        double currentSum = 0.0;
        for (int col = 0; col < matrixSize; col++) {
            currentSum += originalMatrix[row][col] * solutionVector[col];
        }
        double error = abs(currentSum - originalConstants[row]);
        if (error > maxError) {
            maxError = error;
        }
    }
    return maxError;
}

// ==================== MAIN PROGRAM ====================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int processRank, totalProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcesses);

    vector<int> testSizes = { 100, 500, 1000, 2000 };
    vector<ExperimentResult> results;

    if (processRank == 0) {
        cout << "========================================" << endl;
        cout << "Gaussian Elimination Performance Test" << endl;
        cout << "========================================" << endl << endl;
    }

    for (int matrixSize : testSizes) {
        if (processRank == 0) {
            cout << "Matrix Size: " << matrixSize << " x " << matrixSize << endl;
            cout << "----------------------------------------" << endl;
        }

        // Create matrices
        vector<vector<double>> originalMatrix(matrixSize, vector<double>(matrixSize));
        vector<double> originalConstants(matrixSize);

        // Generate system (processRank 0)
        if (processRank == 0) {
            generateSystem(originalMatrix, originalConstants, matrixSize);
        }

        // Broadcast to all processes
        for (int i = 0; i < matrixSize; i++) {
            MPI_Bcast(originalMatrix[i].data(), matrixSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        MPI_Bcast(originalConstants.data(), matrixSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // SERIAL VERSION
        double serialDuration = 0.0;
        double serialError = 0.0;
        if (processRank == 0) {
            vector<vector<double>> serialMatrix = originalMatrix;
            vector<double> serialConstants = originalConstants;
            vector<double> serialSolution(matrixSize);

            double startTime = omp_get_wtime();
            serialGaussianElimination(serialMatrix, serialConstants, matrixSize);
            serialBackSubstitution(serialMatrix, serialConstants, serialSolution, matrixSize);
            serialDuration = omp_get_wtime() - startTime;

            serialError = checkError(originalMatrix, originalConstants, serialSolution, matrixSize);
        }

        // OPENMP VERSION
        double openmpDuration = 0.0;
        double openmpError = 0.0;
        if (processRank == 0) {
            vector<vector<double>> openmpMatrix = originalMatrix;
            vector<double> openmpConstants = originalConstants;
            vector<double> openmpSolution(matrixSize);

            double startTime = omp_get_wtime();
            openmpGaussianElimination(openmpMatrix, openmpConstants, matrixSize);
            openmpBackSubstitution(openmpMatrix, openmpConstants, openmpSolution, matrixSize);
            openmpDuration = omp_get_wtime() - startTime;

            openmpError = checkError(originalMatrix, originalConstants, openmpSolution, matrixSize);
        }

        // MPI VERSION
        vector<vector<double>> mpiMatrix = originalMatrix;
        vector<double> mpiConstants = originalConstants;
        vector<double> mpiSolution(matrixSize);

        MPI_Barrier(MPI_COMM_WORLD);
        double mpiStartTime = MPI_Wtime();

        mpiGaussianElimination(mpiMatrix, mpiConstants, matrixSize, processRank, totalProcesses);
        mpiBackSubstitution(mpiMatrix, mpiConstants, mpiSolution, matrixSize, processRank, totalProcesses);

        MPI_Barrier(MPI_COMM_WORLD);
        double mpiDuration = MPI_Wtime() - mpiStartTime;

        double mpiError = 0.0;
        if (processRank == 0) {
            mpiError = checkError(originalMatrix, originalConstants, mpiSolution, matrixSize);
        }

        // Print results (processRank 0 only)
        if (processRank == 0) {
            int maxThreads = omp_get_max_threads();
            double speedupOpenMP = serialDuration / openmpDuration;
            double speedupMPI = serialDuration / mpiDuration;
            double efficiencyOpenMP = (speedupOpenMP / maxThreads) * 100.0;
            double efficiencyMPI = (speedupMPI / totalProcesses) * 100.0;

            cout << "Serial Time:      " << serialDuration << " seconds" << endl;
            cout << "OpenMP Time:      " << openmpDuration << " seconds (Threads: "
                << maxThreads << ")" << endl;
            cout << "MPI Time:         " << mpiDuration << " seconds (Processes: "
                << totalProcesses << ")" << endl;
            cout << "OpenMP Speedup:   " << speedupOpenMP << "x (Efficiency: "
                << efficiencyOpenMP << "%)" << endl;
            cout << "MPI Speedup:      " << speedupMPI << "x (Efficiency: "
                << efficiencyMPI << "%)" << endl;
            cout << "Serial Error:     " << serialError << endl;
            cout << "OpenMP Error:     " << openmpError << endl;
            cout << "MPI Error:        " << mpiError << endl;
            cout << endl;

            results.push_back({ matrixSize, serialDuration, openmpDuration, mpiDuration, speedupOpenMP, speedupMPI, efficiencyOpenMP, efficiencyMPI });
        }
    }

    if (processRank == 0) {
        cout << "==========================================================================================================================" << endl;
        cout << "                                              Scalability Summary Table" << endl;
        cout << "==========================================================================================================================" << endl;
        cout << left << setw(15) << "Matrix Size" 
             << setw(18) << "Serial Time (s)" 
             << setw(25) << "OpenMP Parallel Time (s)" 
             << setw(22) << "MPI Parallel Time (s)" 
             << setw(18) << "OpenMP Speedup" 
             << setw(15) << "MPI Speedup" 
             << setw(20) << "OpenMP Efficiency" 
             << setw(18) << "MPI Efficiency" << endl;
        cout << "--------------------------------------------------------------------------------------------------------------------------" << endl;
        
        for (const auto& res : results) {
            cout << left << setw(15) << res.matrixSize 
                 << setw(18) << fixed << setprecision(4) << res.serialTime 
                 << setw(25) << res.openmpTime 
                 << setw(22) << res.mpiTime 
                 << setw(18) << res.openmpSpeedup 
                 << setw(15) << res.mpiSpeedup 
                 << setw(20) << res.openmpEfficiency 
                 << setw(18) << res.mpiEfficiency << endl;
        }
        cout << "==========================================================================================================================" << endl;
    }

    MPI_Finalize();
    return 0;
}
