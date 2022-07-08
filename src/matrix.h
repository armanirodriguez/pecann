#pragma once

typedef struct Matrix {
    float *data;
    unsigned rows, cols;
} Matrix;

#define get(m,row,col) (m.data[(row) * (m.cols) + (col)])
#define len(m) (m.rows * m.cols)

Matrix matrix(unsigned rows, unsigned cols);
Matrix matrixFromData(unsigned rows, unsigned cols, float *data);
Matrix copy(Matrix m);
Matrix add(Matrix m1, Matrix m2);
void addInPlace(Matrix m1, Matrix m2);
void subInPlace(Matrix m1, Matrix m2);
Matrix sub(Matrix m1, Matrix m2);
Matrix scalarMult(Matrix m1, float f);
Matrix mult(Matrix m1, Matrix m2);
Matrix hadamard(Matrix m1, Matrix m2);
Matrix transpose(Matrix m);
int maxIndex(Matrix m);
Matrix applyFunc(Matrix m, float (*func)(float));
void applyFuncInPlace(Matrix m, float (*func)(float));
void freeMatrix(Matrix m);
void printMatrix(Matrix m);
void saveMatrixToFile(FILE *file, Matrix m);
Matrix readMatrixFromFile(FILE *file);