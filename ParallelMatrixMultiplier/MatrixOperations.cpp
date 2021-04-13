//
// Created by rey on 4/3/21.
//
#include "MatrixOperations.h"
#include <mpi.h>
#include <cstdlib>

double rand_double(){
    return (double)rand()/RAND_MAX*50.0 - 2.0;
}

void DataInitialization(double* pMatrix, int rowCount, int colCount) {
    for (int i = 0; i < rowCount; i++){
       for (int j = 0; j < colCount; j++){
           pMatrix[i*colCount + j] = rand_double();
       }
   }
}

void SetToZero(double* pMatrix, int rowCount, int colCount) {
    for (int i = 0; i < rowCount; i++){
        for (int j = 0; j < colCount; j++){
            pMatrix[i*colCount + j] = 0;
        }
    }
}

// Function for formatted vector output
void PrintVector(double* pVector, int Size, int ProcNum) {
    printf("proc #%d ", ProcNum);
//    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < Size; i++)
        printf("%7.4f ", pVector[i]);

//    MPI_Barrier(MPI_COMM_WORLD);
    printf("\n");
}

// Function for formatted vector output
void PrintVector(int* pVector, int Size, int ProcNum) {
    printf("proc # %d ", ProcNum);
//    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < Size; i++)
        printf("%d ", pVector[i]);

//    MPI_Barrier(MPI_COMM_WORLD);
    printf("\n");
}

// Function for formatted matrix output
void PrintMatrix(double* pMatrix, int RowCount, int ColCount) {
    int i, j; // Loop variables
    for (i = 0; i < RowCount; i++) {
        for (j = 0; j < ColCount; j++)
            printf("%7.4f ", pMatrix[i * ColCount + j]);
        printf("\n");
    }
}

// Function for matrix multiplication
void MatrixMul(double* pAMatrix, double* pBMatrix, double* pCMatrix, int n1, int n2, int n3) {
    int i, j, k;  // Loop variables
    for (i = 0; i < n1; i++) {
        for (j = 0; j < n3; j++)
            for (k = 0; k < n2; k++)
                pCMatrix[i*n3 + j] += pAMatrix[i*n2 + k] * pBMatrix[k*n3 + j];
    }
}
