#include <iostream>
#include <mpi.h>
#include <cmath>
#include <zconf.h>
#include "MatrixOperations.h"

MPI_Comm GridComm;    // Grid communicator
MPI_Comm ColComm;     // Column communicator
MPI_Comm RowComm;     // Row communicator
int GridCoords[2];
int ProcNum = 0;
int ProcRank = 0;
int p1 = 6;
int p2 = 3;
int n1 = 24;
int n2 = 8;
int n3 = 9;

void CreateGridCommunicators(){
    int DimSize[2]; // Number of processes in each dimension of the grid
    int Periodic[2]; // =1, if the grid dimension should be periodic
    int SubDimension[2];  // =1, if the grid dimension should be fixed

    DimSize[0] = p1;
    DimSize[1] = p2;

    Periodic[0] = 1;
    Periodic[1] = 1;
    // Creation of the Cartesian communicator
    MPI_Cart_create(MPI_COMM_WORLD, 2, DimSize, Periodic, 0, &GridComm);
    // Determination of the cartesian coordinates for every process
    MPI_Cart_coords(GridComm, ProcRank, 2, GridCoords);

    // Creating communicators for rows
    SubDimension[0] = 0; // Dimension is fixed
    SubDimension[1] = 1; // Dimension belong to the subgrid
    MPI_Cart_sub(GridComm, SubDimension, &RowComm);

    // Creating communicators for columns
    SubDimension[0] = 1; // Dimension belong to the subgrid
    SubDimension[1] = 0; // Dimension is fixed
    MPI_Cart_sub(GridComm, SubDimension, &ColComm);
}

void InitializeProcess(double* &pAMatrix, double* &pBMatrix, double* &pCMatrix,
                       double* &pAblock, double* &pBblock, double* &pCblock, int &ABlockSize, int &BBlockSize ) {

    ABlockSize = n1/p1;
    BBlockSize = n3/p2;

    pAblock = new double [n2*ABlockSize];
    pBblock = new double [n2*BBlockSize];
    pCblock = new double [ABlockSize*BBlockSize];

    if (ProcRank == 0){
        pAMatrix = new double [n1*n2];
        pBMatrix = new double [n2*n3];
        pCMatrix = new double [n1*n3];
        DataInitialization(pAMatrix, n1, n2);
        DataInitialization(pBMatrix, n2, n3);
        SetToZero(pCMatrix, n1, n3);
    }

    SetToZero(pCblock, ABlockSize, BBlockSize);
}

void TerminateProcess (double* AMatrix, double* BMatrix,
                       double* CMatrix, double* Ablock, double* Bblock, double* Cblock){

    if (ProcRank == 0){
        delete [] AMatrix;
        delete [] BMatrix;
        delete [] CMatrix;
    }
    delete [] Ablock;
    delete [] Bblock;
    delete [] Cblock;
}

void DataDistribution(double* AMatrix, double* BMatrix, double* Ablock,
                      double* Bblock, int ABlockSize, int BBlockSize) {

    if (GridCoords[1] == 0) {
        MPI_Scatter(AMatrix, ABlockSize * n2, MPI_DOUBLE, Ablock,
                ABlockSize * n2, MPI_DOUBLE, 0, ColComm);
    }

    MPI_Bcast(Ablock, ABlockSize * n2, MPI_DOUBLE, 0, RowComm);
    MPI_Datatype col, coltype;

    MPI_Type_vector(n2, BBlockSize, n3, MPI_DOUBLE, &col);
    MPI_Type_commit(&col);
    MPI_Type_create_resized(col, 0, BBlockSize * sizeof(double), &coltype);
    MPI_Type_commit(&coltype);

    if (GridCoords[0] == 0) {
        MPI_Scatter(BMatrix, 1, coltype, Bblock, n2 * BBlockSize,
                    MPI_DOUBLE, 0, RowComm);
    }

    MPI_Bcast(Bblock, BBlockSize * n2, MPI_DOUBLE, 0, ColComm);
}

int main(int argc, char* argv[]) {
    double* AMatrix = NULL;      // First argument of matrix multiplication
    double* BMatrix = NULL;      // Second argument of matrix multiplication
    double* CMatrix = NULL;      // Result matrix

    int ABlockSize = 0;
    int BBlockSize = 0;

    double *Ablock = NULL;       // Current block of matrix A
    double *Bblock = NULL;       // Current block of matrix B
    double *Cblock = NULL;       // Block of result matrix C

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    if ((n1 % p1 != 0) || (n3 % p2 != 0)) {
        if (ProcRank == 0) {
            printf ("invalid grid size\n");
        }
    } else {
        if (ProcRank == 0)
            printf("Parallel matrix multiplication program, on %d processes\n", ProcNum);
        CreateGridCommunicators(); // Grid communicator creating
    }

    // Memory allocation and initialization of matrix elements
    InitializeProcess(AMatrix, BMatrix, CMatrix, Ablock, Bblock,
                      Cblock, ABlockSize, BBlockSize);

    double startTime;

    if (ProcRank == 0) {
        startTime = MPI_Wtime();
        printf("Initial matrix A \n");
        PrintMatrix(AMatrix, n1, n2);
        printf("Initial matrix B \n");
        PrintMatrix(BMatrix, n2, n3);
    }
    // distribute data among the processes
    DataDistribution(AMatrix, BMatrix, Ablock, Bblock, ABlockSize, BBlockSize);
    // Multiply matrix blocks of the current process
    MatrixMul(Ablock, Bblock, Cblock, ABlockSize, n2, BBlockSize);

    // Gather all data in one matrix

    // the first step is creating a block type and resizing it
    MPI_Datatype block, blocktype;
    MPI_Type_vector(ABlockSize, BBlockSize, n3, MPI_DOUBLE, &block);
    MPI_Type_commit(&block);

    MPI_Type_create_resized(block, 0, BBlockSize*sizeof(double), &blocktype);
    MPI_Type_commit(&blocktype);

    // calculate displ
    int* displ =  new int[p1*p2];
    int* rcount =  new int[p1*p2];
    int BlockCount = 0;
    int BlockSize = ABlockSize*BBlockSize;
    int NumCount = 0;
    int Written;
    int j = 0;

    while (NumCount < p1*p2*BlockSize) {
        Written = 0;
        for (int i = 0; i < n3; i += BBlockSize) {
            displ[j] = BlockCount;
            rcount[j] = 1;
            j++;
            BlockCount++;

            Written++;
        }
        NumCount += Written * BlockSize;
        BlockCount += Written * (ABlockSize - 1);
    }

    MPI_Gatherv(Cblock, BlockSize, MPI_DOUBLE, CMatrix,
                rcount, displ, blocktype, 0, MPI_COMM_WORLD);

    if (ProcRank == 0){
        double endTime = MPI_Wtime();
        printf("matrix C \n");

        PrintMatrix(CMatrix, n1, n3);
        printf("That took %lf seconds\n",endTime-startTime);
    }

    TerminateProcess(AMatrix, BMatrix, CMatrix, Ablock, Bblock, Cblock);
    delete [] displ;
    delete [] rcount;
    MPI_Finalize();
    return 0;
}

