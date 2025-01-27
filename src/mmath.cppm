module;
#include <ostream>
export module mmath;
export import :vector;
export import :matrix;
export import :eigen;

// Define and export operator<< as a free function template
export template<typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, mmath::Vector<T, N> const& vec) {
  os << "(";
  for (std::size_t i = 0; i < N; ++i) {
    os << vec[i];
    if (i + 1 < N) {
      os << ", ";
    }
  }
  os << ")";
  return os;
}

// Overloads for the partial specializations
export template<typename T>
std::ostream& operator<<(std::ostream& os, mmath::Vector<T,2> const& vec) {
  return os << "(" << vec.x << ", " << vec.y << ")";
}

export template<typename T>
std::ostream& operator<<(std::ostream& os, mmath::Vector<T,3> const& vec) {
  return os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
}
