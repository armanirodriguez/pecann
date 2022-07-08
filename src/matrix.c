#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "matrix.h"

Matrix matrix(unsigned rows, unsigned cols) {
    assert(rows > 0 && cols > 0);
    Matrix m = {
        .rows = rows,
        .cols = cols
    };
    m.data = (float*) calloc(len(m), sizeof(float));
    assert(m.data);
    return m;
}

Matrix matrixFromData(unsigned rows, unsigned cols, float *data) {
    assert(rows > 0 && cols > 0 && data);
    Matrix m = {
        .rows = rows,
        .cols = cols
    };
    m.data = data;
    return m;
}

Matrix copy(Matrix m) {
    float *orig = m.data;
    m.data = (float*) malloc(len(m) * sizeof(float));
    assert(m.data);
    memcpy(m.data, orig, len(m) * sizeof(float));
    return m;
}

Matrix mult(Matrix m1, Matrix m2) {
    Matrix result = matrix(m1.rows, m2.cols);
    float *ptr = result.data;
    for (unsigned i = 0; i < m1.rows; i++) {
        for (unsigned j = 0; j < m2.cols; j++) {
            for (unsigned k = 0; k < m1.cols; k++) {
                ptr[0] += get(m1,i,k) * get(m2,k,j); 
            }
            ptr++;
        }
    }
    return result;
}

Matrix add(Matrix m1, Matrix m2) {
    assert(m1.rows == m2.rows && m1.cols == m2.cols);
    Matrix result = matrix(m1.rows, m1.cols);
    for (unsigned i = 0; i < len(m1); i++) {
        result.data[i] = m1.data[i] + m2.data[i];
    }
    return result;
}

void addInPlace(Matrix m1, Matrix m2) {
    assert(m1.rows == m2.rows && m1.cols == m2.cols);
    for (unsigned i = 0; i < len(m1); i++) {
        m1.data[i] += m2.data[i];
    }
}

Matrix sub(Matrix m1, Matrix m2) {
    assert(m1.rows == m2.rows && m1.cols == m2.cols);
    Matrix result = matrix(m1.rows, m1.cols);
    for (unsigned i = 0; i < len(m1); i++) {
        result.data[i] = m1.data[i] - m2.data[i];
    }
    return result;
}

void subInPlace(Matrix m1, Matrix m2) {
    assert(m1.rows == m2.rows && m1.cols == m2.cols);
    for (unsigned i = 0; i < len(m1); i++) {
        m1.data[i] -= m2.data[i];
    }
}

Matrix scalarMult(Matrix m1, float f) {
    Matrix result = copy(m1);
    for (unsigned i = 0; i < len(m1); i++) {
        result.data[i] *= f;
    }
    return result;
}

Matrix hadamard(Matrix m1, Matrix m2) {
    assert(m1.cols == m2.cols && m1.rows == m2.rows);
    Matrix result = matrix(m1.rows, m1.cols);
    for (unsigned i = 0; i < len(m1); i++) {
        result.data[i] = m1.data[i] * m2.data[i];
    }
    return result;
}

Matrix transpose(Matrix m) {
    Matrix result = matrix(m.cols,m.rows);
    for (unsigned i = 0; i < m.rows; i++) {
        for (unsigned j = 0; j < m.cols; j++) {
            result.data[j*result.cols + i] = m.data[i*m.cols + j];
        }
    }
    return result;
}

Matrix applyFunc(Matrix m, float (*func)(float)) {
    Matrix result = matrix(m.rows, m.cols);
    for (unsigned i = 0; i < len(m); i++) {
        result.data[i] = func(m.data[i]);
    }
    return result;
}

int maxIndex(Matrix m) {
    int maxIndex = 0;
    for (unsigned i = 0; i < len(m); i++) {
        if (m.data[i] > m.data[maxIndex]) {
            maxIndex = i;
        }
    }
    return maxIndex;
}

void applyFuncInPlace(Matrix m, float (*func)(float)) {
    for (unsigned i = 0; i < len(m); i++) {
        m.data[i] = func(m.data[i]);
    }
}

void freeMatrix(Matrix m) {
    if (m.data){
        free(m.data);
    }
}

void printMatrix(Matrix m) {
    for (unsigned i = 0; i < len(m); i++) {
        printf("%0.3f ",m.data[i]);
        if ((i + 1) % m.cols == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

void saveMatrixToFile(FILE *file, Matrix m) {
    fprintf(file, "%d %d ", m.rows, m.cols);
    for (unsigned i = 0; i < len(m); i++) {
        fprintf(file,  "%f ", m.data[i]);
    }
}

Matrix readMatrixFromFile(FILE *file) {
    unsigned rows, cols;
    assert(fscanf(file, "%d %d", &rows, &cols) == 2);
    Matrix m = matrix(rows,cols);
    for (unsigned i = 0; i < len(m); i++) {
        assert(fscanf(file, "%f", &m.data[i]) == 1);
    }
    return m;
}

