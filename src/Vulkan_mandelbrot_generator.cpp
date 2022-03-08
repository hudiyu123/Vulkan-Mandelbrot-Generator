#include "Vulkan_mandelbrot_generator.hpp"

#include <iostream>
#include <fstream>
#include <cmath>
#include <numeric>
#include <execution>

#ifndef NDEBUG
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDebugUtilsMessengerEXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pMessenger) {
  auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
      vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
  if (func) {
    return func(instance, pCreateInfo, pAllocator, pMessenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDebugUtilsMessengerEXT(
    VkInstance instance,
    VkDebugUtilsMessengerEXT messenger,
    const VkAllocationCallbacks* pAllocator) {
  auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
      vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
  if (func) {
    return func(instance, messenger, pAllocator);
  }
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
  std::cerr << pCallbackData->pMessage << "\n";
  return VK_FALSE;
}
#endif

// Read a file in binary format and store it in a char vector.
std::vector<char> read_file(const std::string& filename) {
  auto file = std::ifstream{filename, std::ios::ate | std::ios::binary};

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file!");
  }
  auto fileSize = file.tellg();
  auto buffer = std::vector<char>(static_cast<std::size_t>(fileSize));

  file.seekg(0);
  file.read(buffer.data(), fileSize);

  file.close();

  return buffer;
}

Vulkan_mandelbrot_generator::Vulkan_mandelbrot_generator(int width, int height)
  : width_{width}, height_{height}, workgroup_size_{8, 8} {}

std::vector<unsigned char> Vulkan_mandelbrot_generator::generate() {
  create_instance();
#ifndef NDEBUG
  setup_debug_utils_messenger();
#endif
  find_physical_device();
  create_device();
  create_buffers();
  create_descriptor_set_layout();
  create_descriptor_sets();
  create_compute_pipeline();
  create_command_buffer();
  submit_command_buffer();
  auto raw_image = fetch_rendered_image();
  cleanup();
  return raw_image;
}

std::vector<unsigned char> Vulkan_mandelbrot_generator::fetch_rendered_image() {
  auto count = 4 * width_ * height_;
  auto mapped_memory = device_.mapMemory(
    storage_buffer_memory_, 0, sizeof(float) * count, {});
  auto data = static_cast<float *>(mapped_memory);

  std::vector<unsigned char> image(count, 0);
  // Transform data from [0.0f, 1.0f] (float) to [0, 255] (unsigned char).
  std::transform(std::execution::par_unseq, data, data + count, image.begin(),
    [](auto value){ return static_cast<unsigned char>(255.0f * value); });

  device_.unmapMemory(storage_buffer_memory_);
  return image;
}

void Vulkan_mandelbrot_generator::cleanup() {
  device_.freeMemory(uniform_buffer_memory_);
  device_.destroyBuffer(uniform_buffer_);
  device_.freeMemory(storage_buffer_memory_);
  device_.destroyBuffer(storage_buffer_);
  device_.destroyPipeline(pipeline_);
  device_.destroyPipelineLayout(pipeline_layout_);
  device_.destroyDescriptorPool(descriptor_pool_);
  device_.destroyDescriptorSetLayout(descriptor_set_layout_);
  device_.destroyCommandPool(command_pool_);
  device_.destroy();
#ifndef NDEBUG
  instance_.destroyDebugUtilsMessengerEXT(debug_utils_messenger_);
#endif
  instance_.destroy();
}

void Vulkan_mandelbrot_generator::create_instance() {
  auto application_info = vk::ApplicationInfo{
    .pApplicationName = "Mandelbrot",
    .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
    .pEngineName = "No Engine",
    .engineVersion = VK_MAKE_VERSION(1, 0, 0),
    .apiVersion = VK_API_VERSION_1_0};

#ifndef NDEBUG
  auto layer_properties = vk::enumerateInstanceLayerProperties();
  auto found_validation_layer = std::ranges::any_of(layer_properties,
    [](const auto& property) {
    return std::strcmp("VK_LAYER_KHRONOS_validation", property.layerName) == 0;
  });
  if (!found_validation_layer) {
    throw std::runtime_error{"Validation layer required, but not available!"};
  }
#endif

  auto layers = get_layers();
  auto extensions = get_extensions();
  auto instance_create_info = vk::InstanceCreateInfo{
    .pApplicationInfo = &application_info,
    .enabledLayerCount = static_cast<uint32_t>(layers.size()),
    .ppEnabledLayerNames = layers.data(),
    .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
    .ppEnabledExtensionNames = extensions.data()};

  instance_ = vk::createInstance(instance_create_info);
}

void Vulkan_mandelbrot_generator::find_physical_device() {
  auto physical_devices = instance_.enumeratePhysicalDevices();
  if (physical_devices.empty()) {
    throw std::runtime_error{"Cannot find any physical devices."};
  }
  // We simply choose the first available physical device.
  physical_device_ = physical_devices.front();
}

void Vulkan_mandelbrot_generator::create_device() {
  queue_family_index_ = get_compute_queue_family_index();

  float queue_priority = 1.0f;
  auto queue_create_info = vk::DeviceQueueCreateInfo{
    .queueFamilyIndex = queue_family_index_,
    .queueCount = 1,
    .pQueuePriorities = &queue_priority};

  auto layers = get_layers();
  auto physical_device_features = vk::PhysicalDeviceFeatures{};
  auto device_create_info = vk::DeviceCreateInfo{
    .queueCreateInfoCount = 1,
    .pQueueCreateInfos = &queue_create_info,
    .enabledLayerCount = static_cast<uint32_t>(layers.size()),
    .ppEnabledLayerNames = layers.data(),
    .pEnabledFeatures = &physical_device_features};

  device_ = physical_device_.createDevice(device_create_info);
  queue_ = device_.getQueue(queue_family_index_, 0);
}

void Vulkan_mandelbrot_generator::create_buffers() {
  std::tie(storage_buffer_, storage_buffer_memory_) = create_buffer(
    sizeof(float) * 4 * width_ * height_,
    vk::BufferUsageFlagBits::eStorageBuffer,
    vk::MemoryPropertyFlagBits::eHostCoherent |
    vk::MemoryPropertyFlagBits::eHostVisible);

  std::tie(uniform_buffer_, uniform_buffer_memory_) = create_buffer(
    sizeof(int) * 2,
    vk::BufferUsageFlagBits::eUniformBuffer,
    vk::MemoryPropertyFlagBits::eHostCoherent |
    vk::MemoryPropertyFlagBits::eHostVisible);

  auto mapped_memory = device_.mapMemory(uniform_buffer_memory_, 0,
    sizeof(int) * 2);
  int ubo[] = {width_, height_};
  std::memcpy(mapped_memory, &ubo, sizeof(int) * 2);
  device_.unmapMemory(uniform_buffer_memory_);
}

void Vulkan_mandelbrot_generator::create_descriptor_set_layout() {
  auto bindings = std::vector<vk::DescriptorSetLayoutBinding>{
    {.binding = 0,
     .descriptorType = vk::DescriptorType::eStorageBuffer,
     .descriptorCount = 1,
     .stageFlags = vk::ShaderStageFlagBits::eCompute},
    {.binding = 1,
     .descriptorType = vk::DescriptorType::eUniformBuffer,
     .descriptorCount = 1,
     .stageFlags = vk::ShaderStageFlagBits::eCompute}};

  auto create_info = vk::DescriptorSetLayoutCreateInfo{
    .bindingCount = static_cast<std::uint32_t>(bindings.size()),
    .pBindings = bindings.data()};

  descriptor_set_layout_ = device_.createDescriptorSetLayout(create_info);
}

void Vulkan_mandelbrot_generator::create_descriptor_sets() {
  auto descriptor_pool_sizes = std::vector<vk::DescriptorPoolSize>{
    {.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1},
    {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1}};

  auto descriptor_pool_create_info = vk::DescriptorPoolCreateInfo{
    .maxSets = 2,
    .poolSizeCount = static_cast<std::uint32_t>(descriptor_pool_sizes.size()),
    .pPoolSizes = descriptor_pool_sizes.data()};

  descriptor_pool_ = device_.createDescriptorPool(descriptor_pool_create_info);

  auto descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo{
    .descriptorPool = descriptor_pool_,
    .descriptorSetCount = 1,
    .pSetLayouts = &descriptor_set_layout_};

  descriptor_sets_ = device_.allocateDescriptorSets(
    descriptor_set_allocate_info);

  auto descriptor_storage_buffer_info = vk::DescriptorBufferInfo{
    .buffer = storage_buffer_,
    .offset = 0,
    .range = sizeof(float) * 4 * width_ * height_};

  auto descriptor_uniform_buffer_info = vk::DescriptorBufferInfo{
    .buffer = uniform_buffer_,
    .offset = 0,
    .range = sizeof(int) * 2};

  auto write_descriptor_sets = std::vector<vk::WriteDescriptorSet>{
    {.dstSet = descriptor_sets_.front(),
     .dstBinding = 0,
     .descriptorCount = 1,
     .descriptorType = vk::DescriptorType::eStorageBuffer,
     .pBufferInfo = &descriptor_storage_buffer_info},
    {.dstSet = descriptor_sets_.front(),
     .dstBinding = 1,
     .descriptorCount = 1,
     .descriptorType = vk::DescriptorType::eUniformBuffer,
     .pBufferInfo = &descriptor_uniform_buffer_info}};

  device_.updateDescriptorSets(write_descriptor_sets, {});
}

void Vulkan_mandelbrot_generator::create_compute_pipeline() {
  auto compute_shader_code = read_file("shaders/comp.spv");
  auto shader_module_create_info = vk::ShaderModuleCreateInfo{
    .codeSize = compute_shader_code.size(),
    .pCode = reinterpret_cast<const uint32_t*>(compute_shader_code.data())};

  auto compute_shader_module = device_.createShaderModule(
    shader_module_create_info);

  auto specialization_map_entries = std::vector<vk::SpecializationMapEntry>{
    {.constantID = 0,
     .offset = offsetof(Workgroup_size, x),
     .size = sizeof(std::uint32_t)},
    {.constantID = 1,
     .offset = offsetof(Workgroup_size, y),
     .size = sizeof(std::uint32_t)}};

  auto specialization_info = vk::SpecializationInfo{
    .mapEntryCount = static_cast<std::uint32_t>(
      specialization_map_entries.size()),
    .pMapEntries = specialization_map_entries.data(),
    .dataSize = sizeof(Workgroup_size),
    .pData = &workgroup_size_};

  auto shader_stage_create_info = vk::PipelineShaderStageCreateInfo{
    .stage = vk::ShaderStageFlagBits::eCompute,
    .module = compute_shader_module,
    .pName = "main",
    .pSpecializationInfo = &specialization_info};

  auto pipeline_layout_create_info = vk::PipelineLayoutCreateInfo{
    .setLayoutCount = 1,
    .pSetLayouts = &descriptor_set_layout_};

  pipeline_layout_ = device_.createPipelineLayout(pipeline_layout_create_info);

  auto compute_pipeline_create_info = vk::ComputePipelineCreateInfo{
    .stage = shader_stage_create_info,
    .layout = pipeline_layout_};

  auto create_infos = std::vector<vk::ComputePipelineCreateInfo>{
    compute_pipeline_create_info};
  pipeline_ = device_.createComputePipelines({}, create_infos).value.front();

  device_.destroyShaderModule(compute_shader_module);
}

void Vulkan_mandelbrot_generator::create_command_buffer() {
  auto command_pool_create_info = vk::CommandPoolCreateInfo{
    .queueFamilyIndex = queue_family_index_};

  command_pool_ = device_.createCommandPool(command_pool_create_info);

  auto command_buffer_allocate_info = vk::CommandBufferAllocateInfo{
    .commandPool = command_pool_,
    .level = vk::CommandBufferLevel::ePrimary,
    .commandBufferCount = 1};

  command_buffer_ = device_.allocateCommandBuffers(command_buffer_allocate_info)
    .front();

  auto command_buffer_begin_info = vk::CommandBufferBeginInfo{
    .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};

  command_buffer_.begin(command_buffer_begin_info);
  command_buffer_.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
  command_buffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
    pipeline_layout_, 0, descriptor_sets_, {});
  auto num_workgroup_x = static_cast<std::uint32_t>(std::ceil(
    static_cast<float>(width_) / static_cast<float>(workgroup_size_.x)));
  auto num_workgroup_y = static_cast<std::uint32_t>(std::ceil(
    static_cast<float>(height_) / static_cast<float>(workgroup_size_.y)));
  command_buffer_.dispatch(num_workgroup_x, num_workgroup_y, 1);
  command_buffer_.end();
}

void Vulkan_mandelbrot_generator::submit_command_buffer() {
  auto submit_infos = std::vector<vk::SubmitInfo>{
    {.commandBufferCount = 1,
     .pCommandBuffers = &command_buffer_}};

  auto fence_create_info = vk::FenceCreateInfo{};
  auto fence = device_.createFence(fence_create_info);

  queue_.submit(submit_infos, fence);
  if (device_.waitForFences(1, &fence, VK_TRUE,
    std::numeric_limits<std::uint32_t>::max()) != vk::Result::eSuccess) {
    throw std::runtime_error{"Failed to wait for fence."};
  }

  device_.destroyFence(fence);
}

std::vector<const char*> Vulkan_mandelbrot_generator::get_layers() {
  std::vector<const char*> layers;
#ifndef NDEBUG
  layers.push_back("VK_LAYER_KHRONOS_validation");
#endif
  return layers;
}

std::vector<const char*> Vulkan_mandelbrot_generator::get_extensions() {
  std::vector<const char*> extensions;
#ifndef NDEBUG
  extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif
  return extensions;
}

std::uint32_t Vulkan_mandelbrot_generator::get_compute_queue_family_index() {
  auto queue_family_properties = physical_device_.getQueueFamilyProperties();

  for (std::uint32_t i = 0; i < queue_family_properties.size(); ++i) {
    auto property = queue_family_properties[i];
    if (property.queueCount > 0 &&
      (property.queueFlags & vk::QueueFlagBits::eCompute)) {
      return i;
    }
  }

  throw std::runtime_error(
    "could not find a queue family that supports operations");
}

std::uint32_t Vulkan_mandelbrot_generator::find_memory_type(
  std::uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
  auto memory_properties = physical_device_.getMemoryProperties();

  for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) &&
      (memory_properties.memoryTypes[i].propertyFlags & properties)
      == properties) {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type!");
}

std::pair<vk::Buffer, vk::DeviceMemory>
Vulkan_mandelbrot_generator::create_buffer(vk::DeviceSize size,
  vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties) {
  auto buffer_create_info = vk::BufferCreateInfo{
    .size = size,
    .usage = usage,
    .sharingMode = vk::SharingMode::eExclusive};

  auto buffer = device_.createBuffer(buffer_create_info);

  auto buffer_memory_requirements = device_.getBufferMemoryRequirements(buffer);
  auto allocInfo = vk::MemoryAllocateInfo{
    .allocationSize = buffer_memory_requirements.size,
    .memoryTypeIndex = find_memory_type(
      buffer_memory_requirements.memoryTypeBits, properties)};

  auto buffer_memory = device_.allocateMemory(allocInfo);
  device_.bindBufferMemory(buffer, buffer_memory, 0);

  return {buffer, buffer_memory};
}

#ifndef NDEBUG
void Vulkan_mandelbrot_generator::setup_debug_utils_messenger() {
  auto severity_flags = vk::DebugUtilsMessageSeverityFlagsEXT{
    vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eError};

  auto message_type_flags = vk::DebugUtilsMessageTypeFlagsEXT{
    vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
      vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
      vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation};

  auto create_info = vk::DebugUtilsMessengerCreateInfoEXT{
    .messageSeverity = severity_flags,
    .messageType = message_type_flags,
    .pfnUserCallback = debugCallback};

  debug_utils_messenger_ = instance_.createDebugUtilsMessengerEXT(create_info);
}
#endif