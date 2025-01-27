module;
#include <iostream>
export module mmath:vector;

export namespace mmath{
    template<typename T, std::size_t N>
    class Vector {
      protected:
      std::array<T, N> _data;

      public:

      Vector() { _data.fill({}); }

      // Allow construction from a braced list
      Vector(std::initializer_list<T> init) {
        if (init.size() != N)
          throw std::runtime_error("Vector dimension mismatch");
        std::copy(init.begin(), init.end(), _data.begin());
      }

      // Element access
      T& operator[](std::size_t i) { return _data[i]; }
      const T& operator[](std::size_t i) const { return _data[i]; }

      // Vector addition
      Vector operator+(Vector const& other) const {
        Vector result;
        for (std::size_t i = 0; i < N; ++i) {
            result[i] = _data[i] + other[i];
        }
        return result;
      }
    };

    // Partial specialization for 2D, adding x and y references
    template<typename T>
    class Vector<T, 2> {
      public:
      Vector() : x(T{}), y(T{}) {}
      Vector(T xVal, T yVal) : x(xVal), y(yVal) {}

      // Element-wise addition
      Vector operator+(Vector const& other) const {
        return { x + other.x, y + other.y };
      }

      // Named components
      T x, y;
    };

    // Partial specialization for 3D, adding x, y, and z references
    template<typename T>
    class Vector<T, 3> {
      public:
      Vector() : x(T{}), y(T{}), z(T{}) {}
      Vector(T xVal, T yVal, T zVal) : x(xVal), y(yVal), z(zVal) {}

      Vector operator+(Vector const& other) const {
          return { x + other.x, y + other.y, z + other.z };
      }

      T x, y, z;
    };
}
