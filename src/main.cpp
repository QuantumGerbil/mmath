#include <iostream>
import mmath;

int main() {
    mmath::vector::Vector<float, 3> v1{1.0f, 2.0f, 3.0f};
    mmath::vector::Vector<float, 3> v2{4.0f, 5.0f, 6.0f};
    auto v3 = v1 + v2;
    std::cout << "3D Vector v1: " << v1 << "\n";
    std::cout << "3D Vector v2: " << v2 << "\n";
    std::cout << "v1 + v2 = " << v3 << "\n\n";

    // Using the 4D partial specialization
    mmath::vector::Vector<double, 4> v4{1.1, 2.2, 3.3, 4.4};
    mmath::vector::Vector<double, 4> v5{5.5, 6.6, 7.7, 8.8};
    auto v6 = v4 + v5;
    std::cout << "4D Vector v4: " << v4 << "\n";
    std::cout << "4D Vector v5: " << v5 << "\n";
    std::cout << "v4 + v5 = " << v6 << "\n\n";

    // Using the primary template for 5D
    mmath::vector::Vector<int, 5> v7;
    v7[0] = 1;
    v7[1] = 2;
    v7[2] = 3;
    v7[3] = 4;
    v7[4] = 5;

    mmath::vector::Vector<int, 5> v8;
    v8[0] = 10;
    v8[1] = 20;
    v8[2] = 30;
    v8[3] = 40;
    v8[4] = 50;

    auto v9 = v7 + v8;
    std::cout << "5D Vector v7: " << v7 << "\n";
    std::cout << "5D Vector v8: " << v8 << "\n";
    std::cout << "v7 + v8 = " << v9 << "\n";

    return 0;
}
