#ifndef VULKAN_MANDELBROT_GENERATOR_HPP_
#define VULKAN_MANDELBROT_GENERATOR_HPP_

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>

#include <vector>
#include <cstdint>

class Vulkan_mandelbrot_generator {
 public:
  // Create a generator with the width and the height of the image.
  Vulkan_mandelbrot_generator(int width, int height);

  // Return the raw data of mandelbrot image.
  std::vector<unsigned char> generate();

 private:
  int width_;
  int height_;

  vk::Instance instance_{};
  vk::PhysicalDevice physical_device_{};
  vk::Device device_{};
  vk::Queue queue_{};
  std::uint32_t queue_family_index_{};

  vk::Buffer storage_buffer_{};
  vk::DeviceMemory storage_buffer_memory_{};
  vk::Buffer uniform_buffer_{};
  vk::DeviceMemory uniform_buffer_memory_{};

  vk::DescriptorSetLayout descriptor_set_layout_{};
  vk::DescriptorPool descriptor_pool_{};
  std::vector<vk::DescriptorSet> descriptor_sets_{};

  vk::PipelineLayout pipeline_layout_{};
  vk::Pipeline pipeline_{};

  vk::CommandPool command_pool_{};
  vk::CommandBuffer command_buffer_{};

  void setup();
  std::vector<unsigned char> fetch_rendered_image();
  void cleanup();

  void create_instance();
  void find_physical_device();
  void create_device();
  void create_buffers();
  void create_descriptor_set_layout();
  void create_descriptor_sets();
  void create_compute_pipeline();
  void create_command_buffer();
  void submit_command_buffer();

  static std::vector<const char*> get_layers();
  static std::vector<const char*> get_extensions();
  std::uint32_t get_compute_queue_family_index();
  std::uint32_t find_memory_type(std::uint32_t typeFilter, vk::MemoryPropertyFlags properties);
  std::pair<vk::Buffer, vk::DeviceMemory> create_buffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
    vk::MemoryPropertyFlags properties);

#ifndef NDEBUG
  vk::DebugUtilsMessengerEXT debug_utils_messenger_{};

  void setup_debug_utils_messenger();
#endif

};

#endif //VULKAN_MANDELBROT_GENERATOR_HPP_
