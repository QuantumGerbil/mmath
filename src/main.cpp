#include <iostream>
import mmath;

using mmath::Vector;
using mmath::Matrix;
using mmath::EigenSolver;

bool isSymmetric(const mmath::Matrix<double>& mat) {
    if (mat.getRows() != mat.getCols()) return false;
    size_t n = mat.getRows();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < i; ++j) {
            if (std::abs(mat(i, j) - mat(j, i)) > 1e-10) {
                return false;
            }
        }
    }
    return true;
}


int main() {
    Vector<float, 3> v1{1.0f, 2.0f, 3.0f};
    Vector<float, 3> v2{4.0f, 5.0f, 6.0f};
    auto v3 = v1 + v2;
    std::cout << "3D Vector v1: " << v1 << "\n";
    std::cout << "3D Vector v2: " << v2 << "\n";
    std::cout << "v1 + v2 = " << v3 << "\n\n";

    // Using the 4D partial specialization
    Vector<double, 4> v4{1.1, 2.2, 3.3, 4.4};
    Vector<double, 4> v5{5.5, 6.6, 7.7, 8.8};
    auto v6 = v4 + v5;
    std::cout << "4D Vector v4: " << v4 << "\n";
    std::cout << "4D Vector v5: " << v5 << "\n";
    std::cout << "v4 + v5 = " << v6 << "\n\n";

    // Using the primary template for 5D
    Vector<int, 5> v7;
    v7[0] = 1;
    v7[1] = 2;
    v7[2] = 3;
    v7[3] = 4;
    v7[4] = 5;

    Vector<int, 5> v8;
    v8[0] = 10;
    v8[1] = 20;
    v8[2] = 30;
    v8[3] = 40;
    v8[4] = 50;

    auto v9 = v7 + v8;
    std::cout << "5D Vector v7: " << v7 << "\n";
    std::cout << "5D Vector v8: " << v8 << "\n";
    std::cout << "v7 + v8 = " << v9 << "\n";

    // Create a 3x3 matrix initialized with 0
        Matrix<int> mat1(3, 3, 0);

        // Set some values in the matrix
        mat1(0, 0) = 1;
        mat1(1, 1) = 2;
        mat1(2, 2) = 3;

        std::cout << "Matrix mat1:\n";
        mat1.print();

        // Create another matrix of the same size
        Matrix<int> mat2(3, 3, 5);

        std::cout << "\nMatrix mat2:\n";
        mat2.print();

        // Add the two matrices
        auto mat3 = mat1 + mat2;

        std::cout << "\nMatrix mat3 (mat1 + mat2):\n";
        mat3.print();

        // Multiply a matrix by a scalar
        auto mat4 = mat2 * 2;

        std::cout << "\nMatrix mat4 (mat2 * 2):\n";
        mat4.print();

	Matrix<double> mat(3, 3);
	mat(0, 0) = 4.0; mat(0, 1) = -2.0; mat(0, 2) = 1.0;
	mat(1, 0) = -2.0; mat(1, 1) = 4.0; mat(1, 2) = -2.0;
	mat(2, 0) = 1.0; mat(2, 1) = -2.0; mat(2, 2) = 3.0;

	mat.print();

	if (isSymmetric(mat))
	std::cout << "Is symmetric" << std::endl;
	else
	std::cout << "Not symmetric" << std::endl;

	mmath::EigenSolver<double> solver(mat);

	auto [eigenvalues, eigenvectors] = solver.computeEigenvectors();

	std::cout << "Eigenvalues:\n";
	for (const auto& val : eigenvalues)
        	std::cout << val << "\n";

	std::cout << "\nEigenvectors:\n";
	eigenvectors.print();

    return 0;
}
