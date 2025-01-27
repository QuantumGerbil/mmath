module;
#include <vector>
#include <cmath>
#include <stdexcept>
export module mmath:eigen;

import :matrix;

export namespace mmath {
template <typename T>
class EigenSolver {
private:
    Matrix<T> A;	// The matrix for which we compute eigenvalues/eigenvectors
    size_t maxIterations;
    T tolerance;
    Matrix<T> Q;       // Orthogonal matrix from QR decomposition
    Matrix<T> R;       // Upper triangular matrix from QR decomposition

    // Perform QR decomposition of a square matrix (A = Q * R)
    void qrDecompose(const Matrix<T>& mat, Matrix<T>& Q, Matrix<T>& R) {
        size_t n = mat.getRows();
        Q = Matrix<T>(n, n, 0);
        R = mat;

        // Start with the identity matrix for Q
        for (size_t i = 0; i < n; ++i) {
            Q(i, i) = 1;
        }

        // Gram-Schmidt process to compute Q and R
        for (size_t k = 0; k < n; ++k) {
            T norm = 0;
            for (size_t i = 0; i < n; ++i) {
                norm += R(i, k) * R(i, k);
            }
            norm = std::sqrt(norm);

            // Normalize column k of R to get the k-th column of Q
            for (size_t i = 0; i < n; ++i) {
                Q(i, k) = R(i, k) / norm;
            }

            // Update R using orthogonality
            for (size_t j = k + 1; j < n; ++j) {
                T dotProduct = 0;
                for (size_t i = 0; i < n; ++i) {
                    dotProduct += Q(i, k) * R(i, j);
                }
                for (size_t i = 0; i < n; ++i) {
                    R(i, j) -= dotProduct * Q(i, k);
                }
            }
        }
    }

public:
    // Constructor to initialize with a matrix
    EigenSolver(const Matrix<T>& matrix, size_t maxIterations = 1000, T tolerance = 1e-10)
        : A(matrix), maxIterations(maxIterations), tolerance(tolerance), 
          Q(matrix.getRows(), matrix.getCols(), 0), 
          R(matrix.getRows(), matrix.getCols(), 0) {}

    // Compute eigenvalues using the QR algorithm
    std::vector<T> computeEigenvalues() {
        size_t n = A.getRows();
        if (A.getCols() != n) {
            throw std::invalid_argument("Matrix must be square to compute eigenvalues.");
        }

        Matrix<T> Ak = A;
        for (size_t iter = 0; iter < maxIterations; ++iter) {
            qrDecompose(Ak, Q, R);   // Perform QR decomposition
            Ak = R * Q;              // Update Ak as R * Q

            // Check convergence by examining off-diagonal elements
            bool converged = true;
            for (size_t i = 1; i < n && converged; ++i) {
                for (size_t j = 0; j < i && converged; ++j) {
                    if (std::abs(Ak(i, j)) > tolerance) {
                        converged = false;
                    }
                }
            }
            if (converged) break;
        }

        // Extract eigenvalues from the diagonal of Ak
        std::vector<T> eigenvalues(n);
        for (size_t i = 0; i < n; ++i) {
            eigenvalues[i] = Ak(i, i);
        }
        return eigenvalues;
    }

    // Compute both eigenvalues and eigenvectors
    std::pair<std::vector<T>, Matrix<T>> computeEigenvectors() {
        size_t n = A.getRows();
        if (A.getCols() != n) {
            throw std::invalid_argument("Matrix must be square to compute eigenvectors.");
        }

        Matrix<T> Ak = A;
        Matrix<T> Q_total(n, n, 0);   // Accumulate all Q matrices
        for (size_t i = 0; i < n; ++i) {
            Q_total(i, i) = 1;
        }

        for (size_t iter = 0; iter < maxIterations; ++iter) {
            qrDecompose(Ak, Q, R);   // Perform QR decomposition
            Ak = R * Q;              // Update Ak as R * Q
            Q_total = Q_total * Q;   // Accumulate orthogonal transformations

            // Check convergence by examining off-diagonal elements
            bool converged = true;
            for (size_t i = 1; i < n && converged; ++i) {
                for (size_t j = 0; j < i && converged; ++j) {
                    if (std::abs(Ak(i, j)) > tolerance) {
                        converged = false;
                    }
                }
            }
            if (converged) break;
        }

        // Extract eigenvalues from the diagonal of Ak
        std::vector<T> eigenvalues(n);
        for (size_t i = 0; i < n; ++i) {
            eigenvalues[i] = Ak(i, i);
        }

        return {eigenvalues, Q_total};
    }
};
}
