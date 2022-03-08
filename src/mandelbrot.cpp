#include "Vulkan_mandelbrot_generator.hpp"
#include "Openmp_mandelbrot_generator.hpp"

#include <iostream>
#include "lodepng.h"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Two parameters required." << "\n";
    return 1;
  }
  try {
    int width = std::stoi(argv[1]);
    int height = std::stoi(argv[2]);
    int flag = 0;
    if (argc > 3) {
      flag = std::stoi(argv[3]);
    }

    auto raw_image = std::vector<unsigned char>{};
    if (flag) {
      raw_image = Vulkan_mandelbrot_generator{width, height}.generate();
    } else {
      raw_image = Openmp_mandelbrot_generator{width, height}.generate();
    }

    std::vector<unsigned char> png;
    unsigned int error = lodepng::encode(png, raw_image, width, height);
    if (error) {
      throw std::runtime_error{
        "Failed to encode image" + std::string{lodepng_error_text(error)}};
    }
    std::cout.write(reinterpret_cast<const char*>(png.data()),
      static_cast<std::streamsize>(png.size()));
    return std::cout ? 0 : 1;
  } catch (const vk::SystemError& e) {
    std::cerr << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "unknown exception\n";
    return 1;
  }
}
