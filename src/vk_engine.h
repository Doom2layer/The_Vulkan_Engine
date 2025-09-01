// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>
#include "vk_loader.h"
#include "vk_descriptors.h"

struct DeletionQueue
{
	std::deque<std::function<void()>> deletors;

	void push_function(std::function<void()>&& function)
	{
		deletors.push_back(std::move(function));
	}

	void flush()
	{
		for (auto it = deletors.rbegin(); it != deletors.rend(); ++it)
		{
			(*it)();
		}
		deletors.clear();
	}
};

struct FrameData
{
	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;
	VkSemaphore _swapchainSemaphore, _renderSemaphore;
	VkFence _renderFence;
	DeletionQueue _deletionQueue;
	
};

struct ComputePushedConstants
{
	glm::vec4 data1;
	glm::vec4 data2;
	glm::vec4 data3;
	glm::vec4 data4;
};

struct ComputeEffect
{
	const char* name;

	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;

	ComputePushedConstants constants;
};

constexpr unsigned int FRAME_OVERLAP = 2; // Number of frames we will be "inside" simultaneously

class VulkanEngine {
public:

	bool _isInitialized{ false };
	int _frameNumber {0};
	bool stop_rendering{ false };
	VkExtent2D _windowExtent{ 1700 , 900 };

	struct SDL_Window* _window{ nullptr };

	static VulkanEngine& Get();

	VkInstance _instance; // Vulkan library handle
	VkDebugUtilsMessengerEXT _debugMessenger; // Debug messenger for Vulkan
	VkPhysicalDevice _chosenGPU; // The GPU we will use for rendering
	VkDevice _device; // The logical device we will use for rendering
	VkSurfaceKHR _surface; // The surface we will render to

	VkSwapchainKHR _swapchain;
	VkFormat _swapchainImageFormat;
	
	std::vector<VkImage> _swapchainImages;
	std::vector<VkImageView> _swapchainImageViews;
	VkExtent2D _swapchainExtent;

	DeletionQueue _mainDeletionQueue;

	VmaAllocator _allocator;

	AllocatedImage _drawImage;
	AllocatedImage _depthImage;
	VkExtent2D _drawExtent;
	float render_scale{1.0f};

	DescriptorAllocator global_descriptor_allocator;

	VkDescriptorSet _drawImageDescriptorSet;
	VkDescriptorSetLayout _drawImageDescriptorSetLayout;

	VkPipeline	_gradientPipeline;
	VkPipelineLayout _gradientPipelineLayout;

	VkFence _immFence;
	VkCommandBuffer _immCommandBuffer;
	VkCommandPool _immCommandPool;
	
	FrameData _frames[FRAME_OVERLAP];

	FrameData& get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; };
	
	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamily;

	std::vector<ComputeEffect> backgroundEffects;

	VkPipelineLayout _trianglePipelineLayout;
	VkPipeline _trianglePipeline;

	VkPipelineLayout _meshPipelineLayout;
	VkPipeline _meshPipeline;

	GPUMeshBuffers rectangle;
	
	std::vector<std::shared_ptr<MeshAsset>> meshes;
	
	int currentBackgroundEffect{0};

	bool resize_requested{false};
	
	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();

	//draw background
	void draw_background(VkCommandBuffer cmd);

	//run main loop
	void run();

	// submit a command buffer for immediate submission
	void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& func);

	// draw_imgui
	void draw_imgui(VkCommandBuffer cmd, VkImageView target_image_view);

	//draw geomtry
	void draw_geometry(VkCommandBuffer cmd);
	
	AllocatedBuffer create_buffer(size_t size, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
	void destroy_buffer(const AllocatedBuffer& buffer);

	GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

	void init_mesh_pipeline();
	
private:
	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_sync_structures();
	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();
	void init_descriptors();
	void init_pipelines();
	void init_background_pipelines();
	void init_imgui();
	void init_default_data();
	void resize_swapchain();
};

