#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>

#include "lodepng.h"

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

std::vector<char> readFile(const std::string& filename) {
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

class Mandelbrot {
 private:
  int width_;
  int height_;

  vk::Instance instance_{};

#ifndef NDEBUG
  vk::DebugUtilsMessengerEXT debugUtilsMessenger_{};
#endif

  vk::PhysicalDevice physicalDevice_{};
  vk::Device device_{};
  vk::Queue queue_{};
  std::uint32_t queueFamilyIndex_{};

  vk::Buffer storageBuffer_{};
  vk::DeviceMemory storageBufferMemory_{};
  vk::Buffer uniformBuffer_{};
  vk::DeviceMemory uniformBufferMemory_{};

  vk::DescriptorSetLayout descriptorSetLayout_{};
  vk::DescriptorPool descriptorPool_{};
  std::vector<vk::DescriptorSet> descriptorSets_{};

  vk::PipelineLayout pipelineLayout_{};
  vk::Pipeline pipeline_{};

  vk::CommandPool commandPool_{};
  vk::CommandBuffer commandBuffer_{};

 public:
  Mandelbrot(int width, int height) : width_{width}, height_{height} {}

  std::vector<unsigned char> generate() {
    createInstance();
#ifndef NDEBUG
    setupDebugMessenger();
#endif
    findPhysicalDevice();
    createDevice();
    createBuffers();
    createDescriptorSetLayout();
    createDescriptorSets();
    createComputePipeline();
    createCommandBuffer();
    submitCommandBuffer();
    auto image = fetchRenderedImage();
    cleanup();
    return image;
  };

 private:
  void cleanup() {
    device_.freeMemory(uniformBufferMemory_);
    device_.destroyBuffer(uniformBuffer_);
    device_.freeMemory(storageBufferMemory_);
    device_.destroyBuffer(storageBuffer_);
    device_.destroyPipeline(pipeline_);
    device_.destroyPipelineLayout(pipelineLayout_);
    device_.destroyDescriptorPool(descriptorPool_);
    device_.destroyDescriptorSetLayout(descriptorSetLayout_);
    device_.destroyCommandPool(commandPool_);
    device_.destroy();
#ifndef NDEBUG
    instance_.destroyDebugUtilsMessengerEXT(debugUtilsMessenger_);
#endif
    instance_.destroy();
  }

  std::vector<unsigned char> fetchRenderedImage() {
    auto count = 4 * width_ * height_;
    auto mappedMemory = device_.mapMemory(storageBufferMemory_, 0, sizeof(float) * count, {});
    auto data = static_cast<float *>(mappedMemory);

    std::vector<unsigned char> image(count, 0);
    std::transform(std::execution::par_unseq, data, data + count, image.begin(), [](auto value){
      return static_cast<unsigned char>(255.0f * value);
    });

    device_.unmapMemory(storageBufferMemory_);
    return image;
  }

  void createInstance() {
    auto appInfo = vk::ApplicationInfo{
      .pApplicationName = "Mandelbrot",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "No Engine",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = VK_API_VERSION_1_0};

#ifndef NDEBUG
    auto layerProperties = vk::enumerateInstanceLayerProperties();
    auto found_validation_layer = std::ranges::any_of(layerProperties, [](const auto& property) {
      return std::strcmp("VK_LAYER_KHRONOS_validation", property.layerName) == 0;
    });
    if (!found_validation_layer) {
      throw std::runtime_error{"Validation layer required, but not available!"};
    }
#endif

    auto layers = getLayers();
    auto extensions = getExtensions();

    auto createInfo = vk::InstanceCreateInfo{
      .pApplicationInfo = &appInfo,
      .enabledLayerCount = static_cast<uint32_t>(layers.size()),
      .ppEnabledLayerNames = layers.data(),
      .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
      .ppEnabledExtensionNames = extensions.data()};

    instance_ = vk::createInstance(createInfo);
  }

#ifndef NDEBUG
  void setupDebugMessenger() {
    auto severityFlags = vk::DebugUtilsMessageSeverityFlagsEXT{
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError};

    auto messageTypeFlags = vk::DebugUtilsMessageTypeFlagsEXT{
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation};

    auto createInfo = vk::DebugUtilsMessengerCreateInfoEXT{
        .messageSeverity = severityFlags,
        .messageType = messageTypeFlags,
        .pfnUserCallback = debugCallback};

    debugUtilsMessenger_ = instance_.createDebugUtilsMessengerEXT(createInfo);
  }
#endif

  void findPhysicalDevice() {
    auto physicalDevices = instance_.enumeratePhysicalDevices();
    if (physicalDevices.empty()) {
      throw std::runtime_error{"Cannot find any physical devices."};
    }
    physicalDevice_ = physicalDevices.front();
  }

  std::uint32_t getComputeQueueFamilyIndex() {
    auto queueFamilyProperties = physicalDevice_.getQueueFamilyProperties();

    for (std::uint32_t i = 0; i < queueFamilyProperties.size(); ++i) {
      auto property = queueFamilyProperties[i];
      if (property.queueCount > 0 && (property.queueFlags & vk::QueueFlagBits::eCompute)) {
        return i;
      }
    }

    throw std::runtime_error("could not find a queue family that supports operations");
  }

  void createDevice() {
    queueFamilyIndex_ = getComputeQueueFamilyIndex();

    float queuePriority = 1.0f;
    auto queueCreateInfo = vk::DeviceQueueCreateInfo{
      .queueFamilyIndex = queueFamilyIndex_,
      .queueCount = 1,
      .pQueuePriorities = &queuePriority};

    auto layers = getLayers();
    auto physicalDeviceFeatures = vk::PhysicalDeviceFeatures{};
    auto createInfo = vk::DeviceCreateInfo{
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queueCreateInfo,
      .enabledLayerCount = static_cast<uint32_t>(layers.size()),
      .ppEnabledLayerNames = layers.data(),
      .pEnabledFeatures = &physicalDeviceFeatures};

    device_ = physicalDevice_.createDevice(createInfo);
    queue_ = device_.getQueue(queueFamilyIndex_, 0);
  }

  void createBuffers() {
    std::tie(storageBuffer_, storageBufferMemory_) = createBuffer(
      sizeof(float) * 4 * width_ * height_,
      vk::BufferUsageFlagBits::eStorageBuffer,
      vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible);

    std::tie(uniformBuffer_, uniformBufferMemory_) = createBuffer(
      sizeof(int) * 2,
      vk::BufferUsageFlagBits::eUniformBuffer,
      vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible);

    auto pMappedMemory = device_.mapMemory(uniformBufferMemory_, 0, sizeof(int) * 2);
    int ubo[] = {width_, height_};
    std::memcpy(pMappedMemory, &ubo, sizeof(int) * 2);
    device_.unmapMemory(uniformBufferMemory_);
  }

  void createDescriptorSetLayout() {
    auto bindings = std::vector<vk::DescriptorSetLayoutBinding>{
      {.binding = 0,
       .descriptorType = vk::DescriptorType::eStorageBuffer,
       .descriptorCount = 1,
       .stageFlags = vk::ShaderStageFlagBits::eCompute},
      {.binding = 1,
       .descriptorType = vk::DescriptorType::eUniformBuffer,
       .descriptorCount = 1,
       .stageFlags = vk::ShaderStageFlagBits::eCompute}};

    auto createInfo = vk::DescriptorSetLayoutCreateInfo{
      .bindingCount = static_cast<std::uint32_t>(bindings.size()),
      .pBindings = bindings.data()};

    descriptorSetLayout_ = device_.createDescriptorSetLayout(createInfo);
  }

  void createDescriptorSets() {
    auto descriptorPoolSizes = std::vector<vk::DescriptorPoolSize>{
      {.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1},
      {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1}};

    auto descriptorPoolCreateInfo = vk::DescriptorPoolCreateInfo{
      .maxSets = 2,
      .poolSizeCount = static_cast<std::uint32_t>(descriptorPoolSizes.size()),
      .pPoolSizes = descriptorPoolSizes.data()};

    descriptorPool_ = device_.createDescriptorPool(descriptorPoolCreateInfo);

    auto allocateInfo = vk::DescriptorSetAllocateInfo{
      .descriptorPool = descriptorPool_,
      .descriptorSetCount = 1,
      .pSetLayouts = &descriptorSetLayout_};

    descriptorSets_ = device_.allocateDescriptorSets(allocateInfo);

    auto descriptorStorageBufferInfo = vk::DescriptorBufferInfo{
      .buffer = storageBuffer_,
      .offset = 0,
      .range = sizeof(float) * 4 * width_ * height_};

    auto descriptorUniformBufferInfo = vk::DescriptorBufferInfo{
      .buffer = uniformBuffer_,
      .offset = 0,
      .range = sizeof(int) * 2};

    auto writeDescriptorSets = std::vector<vk::WriteDescriptorSet>{
      {.dstSet = descriptorSets_.front(),
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .pBufferInfo = &descriptorStorageBufferInfo},
      {.dstSet = descriptorSets_.front(),
        .dstBinding = 1,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .pBufferInfo = &descriptorUniformBufferInfo}};

    device_.updateDescriptorSets(writeDescriptorSets, {});
  }

  void createComputePipeline() {
    auto computeShaderCode = readFile("shaders/comp.spv");
    auto shaderModuleCreateInfo = vk::ShaderModuleCreateInfo{
      .codeSize = computeShaderCode.size(),
      .pCode = reinterpret_cast<const uint32_t*>(computeShaderCode.data())};

    auto computeShaderModule = device_.createShaderModule(shaderModuleCreateInfo);

    struct SpecializationData {
      std::uint32_t workGroupSizeX;
      std::uint32_t workGroupSizeY;
    };

    auto specializationMapEntries = std::vector<vk::SpecializationMapEntry>{
      {.constantID = 0, .offset = offsetof(SpecializationData, workGroupSizeX), .size = sizeof(std::uint32_t)},
      {.constantID = 1, .offset = offsetof(SpecializationData, workGroupSizeY), .size = sizeof(std::uint32_t)},
    };

    auto specializationData = SpecializationData{8, 8};

    auto specializationInfo = vk::SpecializationInfo{
      .mapEntryCount = static_cast<std::uint32_t>(specializationMapEntries.size()),
      .pMapEntries = specializationMapEntries.data(),
      .dataSize = sizeof(SpecializationData),
      .pData = &specializationData};

    auto shaderStageCreateInfo = vk::PipelineShaderStageCreateInfo{
      .stage = vk::ShaderStageFlagBits::eCompute,
      .module = computeShaderModule,
      .pName = "main",
      .pSpecializationInfo = &specializationInfo};

    auto layoutCreateInfo = vk::PipelineLayoutCreateInfo{
      .setLayoutCount = 1,
      .pSetLayouts = &descriptorSetLayout_};

    pipelineLayout_ = device_.createPipelineLayout(layoutCreateInfo);

    auto createInfo = vk::ComputePipelineCreateInfo{
      .stage = shaderStageCreateInfo,
      .layout = pipelineLayout_};

    auto createInfos = std::vector<vk::ComputePipelineCreateInfo>{createInfo};

    pipeline_ = device_.createComputePipelines({}, createInfos).value.front();

    device_.destroyShaderModule(computeShaderModule);
  }

  void createCommandBuffer() {
    auto commandPoolCreateInfo = vk::CommandPoolCreateInfo{
      .queueFamilyIndex = queueFamilyIndex_};

    commandPool_ = device_.createCommandPool(commandPoolCreateInfo);

    auto commandBufferAllocateInfo = vk::CommandBufferAllocateInfo{
      .commandPool = commandPool_,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = 1};

    commandBuffer_ = device_.allocateCommandBuffers(commandBufferAllocateInfo).front();

    auto commandBufferBeginInfo = vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};

    commandBuffer_.begin(commandBufferBeginInfo);
    commandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
    commandBuffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout_, 0, descriptorSets_, {});
    commandBuffer_.dispatch(width_, height_, 1);

    commandBuffer_.end();
  }

  void submitCommandBuffer() {
    auto submitInfos = std::vector<vk::SubmitInfo>{{
      .commandBufferCount = 1,
      .pCommandBuffers = &commandBuffer_}};

    auto fenceCreateInfo = vk::FenceCreateInfo{};
    auto fence = device_.createFence(fenceCreateInfo);

    queue_.submit(submitInfos, fence);
    if (device_.waitForFences(1, &fence, VK_TRUE, std::numeric_limits<std::uint32_t>::max()) != vk::Result::eSuccess) {
      throw std::runtime_error{"Failed to wait for fence."};
    }

    device_.destroyFence(fence);
  }

  static std::vector<const char*> getLayers() {
    std::vector<const char*> layers;
#ifndef NDEBUG
    layers.push_back("VK_LAYER_KHRONOS_validation");
#endif
    return layers;
  }

  static std::vector<const char*> getExtensions() {
    std::vector<const char*> extensions;
#ifndef NDEBUG
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif
    return extensions;
  }

  uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    auto memProperties = physicalDevice_.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
        return i;
      }
    }

    throw std::runtime_error("failed to find suitable memory type!");
  }

  std::pair<vk::Buffer, vk::DeviceMemory> createBuffer(
    vk::DeviceSize size,
    vk::BufferUsageFlags usage,
    vk::MemoryPropertyFlags properties) {
    auto bufferInfo = vk::BufferCreateInfo{
      .size = size,
      .usage = usage,
      .sharingMode = vk::SharingMode::eExclusive};

    auto buffer = device_.createBuffer(bufferInfo);

    auto memoryRequirements = device_.getBufferMemoryRequirements(buffer);
    auto allocInfo = vk::MemoryAllocateInfo{
      .allocationSize = memoryRequirements.size,
      .memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, properties)};

    auto bufferMemory = device_.allocateMemory(allocInfo);
    device_.bindBufferMemory(buffer, bufferMemory, 0);

    return {buffer, bufferMemory};
  }
};

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Two parameters required." << "\n";
    return 1;
  }
  try {
    int width = std::stoi(argv[1]);
    int height = std::stoi(argv[2]);

    Mandelbrot mandelbrot{width, height};

    auto image = mandelbrot.generate();
    std::vector<unsigned char> png;
    unsigned error = lodepng::encode(png, image, width, height);
    if (error) {
      throw std::runtime_error{"Failed to encode image" + std::string{lodepng_error_text(error)}};
    }
    std::cout.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
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