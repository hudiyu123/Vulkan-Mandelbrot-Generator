#include <vector>
#include <glm/glm.hpp>
#include <stdexcept>
#include <iostream>

#include "lodepng.h"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Two parameters required." << "\n";
    return 1;
  }

  int width = std::stoi(argv[1]);
  int height = std::stoi(argv[2]);
  int pixel_count = width * height;

  std::vector<glm::vec4> image(pixel_count, glm::vec4{});
#pragma omp parallel for default(none) collapse(2) shared(width, height, image)
  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < height; ++j) {
      float x = static_cast<float>(i) / static_cast<float>(width);
      float y = static_cast<float>(j) / static_cast<float>(height);

      const glm::vec2
        uv = glm::vec2{x, y - 0.5f} * glm::vec2{1.0f, static_cast<float>(height) / static_cast<float>(width)};
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
      image[j * width + i] = glm::vec4(d + e * cos(6.28318f * (f * t + g)), 1.0);
    }
  }

  std::vector<unsigned char> rawImage(pixel_count * 4, 0);
#pragma omp parallel for default(none) shared(pixel_count, image, rawImage)
  for (int i = 0; i < pixel_count; ++i) {
    auto pixel = image[i] * 255.0f;
    auto it = rawImage.begin() + i * 4;
    *it++ = static_cast<unsigned char>(pixel.r);
    *it++ = static_cast<unsigned char>(pixel.g);
    *it++ = static_cast<unsigned char>(pixel.b);
    *it = static_cast<unsigned char>(pixel.a);
  }

  std::vector<unsigned char> pngImage;
  unsigned error = lodepng::encode(pngImage, rawImage, width, height);
  if (error) {
    std::cerr << error << ": " << lodepng_error_text(error) << "\n";
    return 1;
  }
  std::cout.write(reinterpret_cast<const char*>(pngImage.data()), static_cast<std::streamsize>(pngImage.size()));

  return std::cout ? 0 : 1;
}