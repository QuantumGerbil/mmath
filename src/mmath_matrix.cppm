module;
#include <iostream>
#include <vector>
#include <stdexcept>
#include <iomanip>
export module mmath:matrix;

export namespace mmath{
template <typename T>
class Matrix {
private:
    size_t rows, cols;   // Dimensions of the matrix
    std::vector<T> data; // Flattened data storage

public:
    // Constructor: Initialize with rows, cols, and an optional initial value
    Matrix(size_t rows, size_t cols, const T& init = T())
        : rows(rows), cols(cols), data(rows * cols, init) {}

    // Copy constructor
    Matrix(const Matrix& other) = default;

    // Assignment operator
    Matrix& operator=(const Matrix& other) = default;

    // Access element (non-const)
    T& operator()(size_t row, size_t col) {
        if (row >= rows || col >= cols) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return data[row * cols + col];
    }

    // Access element (const)
    const T& operator()(size_t row, size_t col) const {
        if (row >= rows || col >= cols) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return data[row * cols + col];
    }

    // Get the number of rows
    size_t getRows() const { return rows; }

    // Get the number of columns
    size_t getCols() const { return cols; }

    // Addition operator
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    // Subtraction operator
    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must match for subtraction");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    // Multiplication operator (element-wise)
    Matrix operator*(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must match for multiplication");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }

    // Scalar multiplication
    Matrix operator*(const T& scalar) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }

    // Print the matrix
    void print() const {
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                std::cout << std::setw(8) << (*this)(r, c);
            }
            std::cout << '\n';
        }
    }
};
}
