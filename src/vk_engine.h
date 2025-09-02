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
	VkCommandPool command_pool;
	VkCommandBuffer main_command_buffer;
	VkSemaphore swapchain_semaphore, render_semaphore;
	VkFence render_fence;
	DeletionQueue deletion_queue;
	ExtendableDescriptorAllocator frame_descriptor_allocator;
};

struct ComputePushConstants
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
	VkPipelineLayout pipeline_layout;
	ComputePushConstants constants;
};

struct GPUSceneData
{
	glm::mat4 view;
	glm::mat4 projection;
	glm::mat4 view_projection;
	glm::vec4 ambient_color;
	glm::vec4 sunlight_direction;
	glm::vec4 sunlight_color;
};

struct RenderObject
{
	uint32_t index_count;
	uint32_t first_index;
	VkBuffer index_buffer;
	MaterialInstance* material;
	glm::mat4 transform;
	VkDeviceAddress vertex_buffer_address;
};

constexpr unsigned int FRAME_OVERLAP = 2; // Number of frames we will be "inside" simultaneously

class VulkanEngine {
public:
	// Core engine state
	bool is_initialized{false};
	int frame_number{0};
	bool stop_rendering{false};
	bool resize_requested{false};
	
	// Window properties
	VkExtent2D window_extent{1700, 900};
	struct SDL_Window* window{nullptr};
	
	// Core Vulkan objects
	VkInstance instance; // Vulkan library handle
	VkDebugUtilsMessengerEXT debug_messenger; // Debug messenger for Vulkan
	VkPhysicalDevice chosen_gpu; // The GPU we will use for rendering
	VkDevice device; // The logical device we will use for rendering
	VkSurfaceKHR surface; // The surface we will render to

	// Swapchain related
	VkSwapchainKHR swapchain;
	VkFormat swapchain_image_format;
	std::vector<VkImage> swapchain_images;
	std::vector<VkImageView> swapchain_image_views;
	VkExtent2D swapchain_extent;

	// Memory management
	DeletionQueue main_deletion_queue;
	VmaAllocator allocator;

	// Rendering resources
	AllocatedImage draw_image;
	AllocatedImage depth_image;
	VkExtent2D draw_extent;
	float render_scale{1.0f};

	// Descriptors
	DescriptorAllocator global_descriptor_allocator;
	VkDescriptorSet draw_image_descriptor_set;
	VkDescriptorSetLayout draw_image_descriptor_set_layout;

	// Compute pipeline
	VkPipeline gradient_pipeline;
	VkPipelineLayout gradient_pipeline_layout;

	// Immediate submit resources
	VkFence immediate_fence;
	VkCommandBuffer immediate_command_buffer;
	VkCommandPool immediate_command_pool;
	
	// Frame data
	FrameData frames[FRAME_OVERLAP];
	FrameData& get_current_frame() { return frames[frame_number % FRAME_OVERLAP]; }

	// Queue management
	VkQueue graphics_queue;
	uint32_t graphics_queue_family;

	// Effects and pipelines
	std::vector<ComputeEffect> background_effects;
	VkPipelineLayout triangle_pipeline_layout;
	VkPipeline triangle_pipeline;
	VkPipelineLayout mesh_pipeline_layout;
	VkPipeline mesh_pipeline;

	// Mesh data
	GPUMeshBuffers rectangle;
	std::vector<std::shared_ptr<MeshAsset>> meshes;
	int current_background_effect{0};

	// Scene data
	GPUSceneData scene_data;
	VkDescriptorSetLayout gpu_scene_descriptor_set_layout;

	// Texture
	AllocatedImage white_image;
	AllocatedImage black_image;
	AllocatedImage grey_image;
	AllocatedImage error_checkerboard_image;
	VkSampler default_sampler_linear;
	VkSampler default_sampler_nearest;
	VkDescriptorSetLayout single_image_descriptor_set_layout;
	
	// Core engine functions
	void init(); // initializes everything in the engine
	void cleanup(); // shuts down the engine
	void draw(); // draw loop
	void run(); // run main loop

	// Rendering functions
	void draw_background(VkCommandBuffer cmd); // draw background
	void draw_imgui(VkCommandBuffer cmd, VkImageView target_image_view); // draw_imgui
	void draw_geometry(VkCommandBuffer cmd); // draw geometry
	
	// Utility functions
	void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& func); // submit a command buffer for immediate submission
	AllocatedBuffer create_buffer(size_t size, VkBufferUsageFlags usage, VmaMemoryUsage memory_usage);
	void destroy_buffer(const AllocatedBuffer& buffer);
	GPUMeshBuffers upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

	// Pipeline initialization
	void init_mesh_pipeline();

	// Texture
	AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
	AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
	void destroy_image(const AllocatedImage& image);
	
private:
	// Initialization functions
	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_sync_structures();
	void init_descriptors();
	void init_pipelines();
	void init_background_pipelines();
	void init_imgui();
	void init_default_data();
	
	// Swapchain management
	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();
	void resize_swapchain();
};
