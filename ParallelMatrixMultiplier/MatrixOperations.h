//
// Created by rey on 4/3/21.
//

#ifndef PARALLELMATRIXMUL_MATRIXOPERATIONS_H
#define PARALLELMATRIXMUL_MATRIXOPERATIONS_H

void DataInitialization(double* pMatrix, int rowCount, int colCount);
void PrintMatrix(double* pMatrix, int RowCount, int ColCount);
void PrintVector(double* pVector, int Size, int ProcNum);
void SetToZero(double* pMatrix, int rowCount, int colCount);
void MatrixMul(double* pAMatrix, double* pBMatrix, double* pCMatrix,
               int n1, int n2, int n3);
void PrintVector(int* pVector, int Size, int ProcNum);

#endif //PARALLELMATRIXMUL_MATRIXOPERATIONS_H
