#ifndef OPENMP_MANDELBROT_GENERATOR_HPP_
#define OPENMP_MANDELBROT_GENERATOR_HPP_

#include <vector>

class Openmp_mandelbrot_generator {
 public:
  Openmp_mandelbrot_generator(int width, int height);

  std::vector<unsigned char> generate() const;

 private:
  int width_;
  int height_;
};

#endif //OPENMP_MANDELBROT_GENERATOR_HPP_
