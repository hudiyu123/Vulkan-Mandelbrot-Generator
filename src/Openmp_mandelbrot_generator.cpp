#include "Openmp_mandelbrot_generator.hpp"

#include <glm/glm.hpp>

Openmp_mandelbrot_generator::Openmp_mandelbrot_generator(int width, int height) : width_{width}, height_{height} {}

std::vector<unsigned char> Openmp_mandelbrot_generator::generate() const {
  int pixel_count = width_ * height_;
  auto image = std::vector<glm::vec4>(pixel_count, glm::vec4{});

#pragma omp parallel for default(none) collapse(2) shared(width_, height_, image)
  for (int i = 0; i < width_; ++i) {
    for (int j = 0; j < height_; ++j) {
      float x = static_cast<float>(i) / static_cast<float>(width_);
      float y = static_cast<float>(j) / static_cast<float>(height_);

      const glm::vec2
        uv = glm::vec2{x, y - 0.5f} * glm::vec2{1.0f, static_cast<float>(height_) / static_cast<float>(width_)};
      const glm::vec2 c = uv * 3.0f + glm::vec2(-2.1f, 0.0f);
      auto z = glm::vec2{0.0f};
      const int m = 128;
      int n = 0;
      for (int k = 0; k < m; ++k) {
        z = glm::vec2(z.x * z.x - z.y * z.y, 2.0f * z.x * z.y) + c;
        if (glm::dot(z, z) > 4) { break; }
        ++n;
      }

      float t = static_cast<float>(n) / static_cast<float>(m);
      auto d = glm::vec3(0.3, 0.3, 0.5);
      auto e = glm::vec3(-0.2, -0.3, -0.5);
      auto f = glm::vec3(2.1, 2.0, 3.0);
      auto g = glm::vec3(0.0, 0.1, 0.0);
      image[j * width_ + i] = glm::vec4(d + e * cos(6.28318f * (f * t + g)), 1.0);
    }
  }

  std::vector<unsigned char> raw_image(pixel_count * 4, 0);
#pragma omp parallel for default(none) shared(pixel_count, image, raw_image)
  for (int i = 0; i < pixel_count; ++i) {
    auto pixel = image[i] * 255.0f;
    auto it = raw_image.begin() + i * 4;
    *it++ = static_cast<unsigned char>(pixel.r);
    *it++ = static_cast<unsigned char>(pixel.g);
    *it++ = static_cast<unsigned char>(pixel.b);
    *it = static_cast<unsigned char>(pixel.a);
  }

  return raw_image;
}
