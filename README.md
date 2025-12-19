Gaussian Elimination Performance Comparison
Project Overview
This project implements Gaussian elimination with back substitution for solving linear systems of equations using three different parallelization approaches:

Serial Implementation: Standard single-threaded execution
OpenMP: Shared-memory parallelization using multi-threading
MPI: Distributed-memory parallelization using multiple processes

The program benchmarks and compares the performance of each approach across different matrix sizes (100×100, 500×500, 1000×1000, and 2000×2000).
-------------------------------------------------------------------------------------------------------------------------------------------------------
Prerequisites

**OpenMP setup**
C++ Compiler with OpenMP Support
Windows (Visual Studio):

1. Right-click on the GaussianElimination project folder
2. Select Properties
3. Navigate to C/C++ → Language
4. Set Open MP Support to Yes
5. Click Apply and OK
-------------------------------------------------------------------------------------------------------------------------------------------------------
**Microsoft MPI (MS-MPI) Setup**
To run the distributed version, you must link the MS-MPI SDK:

*A. Installation*
    - Download and install msmpisetup.exe and msmpisdk.msi from [Microsoft MPI Downloads](https://www.microsoft.com/en-us/download/details.aspx?id=57467).

*B. Project Configuration*
1. Include Directories: * Properties → C/C++ → General → Additional Include Directories.
   - Add: C:\Program Files (x86)\Microsoft SDKs\MPI\Include
2. Library Directories: * Properties → Linker → General → Additional Library Directories.
   - Add: C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64
3. Linker Input: * Properties → Linker → Input → Additional Dependencies.
   - Add: msmpi.lib
4. Environment Variables: * Ensure C:\Program Files\Microsoft MPI\Bin\ is in your System Path.
    
-------------------------------------------------------------------------------------------------------------------------------------------------------
**HOW TO RUN CODE**
Step 1: Run the project once
Windows (Visual Studio):

Step 2: Open Terminal and Navigate 

Go to View → Terminal
Navigate to the executable location by typing:

cd /d <path-to-project>\x64\Debug
Where <path-to-project> is the full path to your GaussianElimination project folder.
Example:
cd /d C:\Users\USER\source\repos\PDC-Gaussian-Elimination\GaussianEliminationProject\GaussianElimination\x64\Debug
In this example:
<path-to-project> = C:\Users\USER\source\repos\PDC-Gaussian-Elimination\GaussianEliminationProject\GaussianElimination

Or if you're already in the project folder:
cd x64\Debug
Note: The /d flag allows you to change drives if your project is on a different drive (D:, E:, etc.)

**Run with Multiple MPI Processes (Recommended):**
Windows (with 4 processes):
mpiexec -n 4 GaussianElimination.exe
Change the number of processes as needed:
mpiexec -n 2 GaussianElimination.exe   # Run with 2 processes
mpiexec -n 4 GaussianElimination.exe   # Run with 4 processes
mpiexec -n 8 GaussianElimination.exe   # Run with 8 processes
mpiexec -n 16 GaussianElimination.exe  # Run with 16 processes
