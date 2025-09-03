//> includes
#include "vk_engine.h"
#include "vk_images.h"
//bootstrap library
#include "VkBootstrap.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_initializers.h>
#include <vk_types.h>

#include <chrono>
#include <thread>

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#define VMA_IMPLEMENTATION
#include <iostream>
#include <glm/ext/matrix_clip_space.hpp>

#include "vk_mem_alloc.h"
#include "vk_descriptors.h"
#include "vk_pipelines.h"
#include <glm/gtx/transform.hpp>

constexpr bool USE_VALIDATION_LAYERS = false;

void VulkanEngine::init()
{
    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    window = SDL_CreateWindow
    (
        "Mustafa Hazeen Engine",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        window_extent.width,
        window_extent.height,
        window_flags
    );

    init_vulkan();
    init_swapchain();
    init_commands();
    init_sync_structures();
    init_descriptors();
    init_pipelines();
    init_imgui();
    init_default_data();

    // everything went fine
    is_initialized = true;

    editor_camera.velocity = glm::vec3(0.0f);
    editor_camera.position = glm::vec3(0.0f, 0.0f, 5.0f);
    editor_camera.pitch = 0;
    editor_camera.yaw = 0;
}

void VulkanEngine::init_pipelines()
{
    init_background_pipelines();
    init_mesh_pipeline();
    metallic_roughness.build_pipelines(this);
}

void VulkanEngine::cleanup()
{
    if (is_initialized) {
        // make sure the gpu has no processing to do before we start destroying things
        vkDeviceWaitIdle(device);
        
        for (int i = 0; i < FRAME_OVERLAP; i++) {
            vkDestroyCommandPool(device, frames[i].command_pool, nullptr);

            //destroy sync objects
            vkDestroyFence(device, frames[i].render_fence, nullptr);
            vkDestroySemaphore(device, frames[i].render_semaphore, nullptr);
            vkDestroySemaphore(device, frames[i].swapchain_semaphore, nullptr);

            frames[i].deletion_queue.flush();
        }

        for (auto& mesh : meshes)
        {
            destroy_buffer(mesh->mesh_buffers.indexBuffer);
            destroy_buffer(mesh->mesh_buffers.vertexBuffer);
        }

        metallic_roughness.clear_resources(device);
        
        main_deletion_queue.flush();

        destroy_swapchain();

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyDevice(device, nullptr);

        vkb::destroy_debug_utils_messenger(instance, debug_messenger);
        vkDestroyInstance(instance, nullptr);
        
        SDL_DestroyWindow(window);
    }
}

void VulkanEngine::draw()
{
    update_scene();
    // wait until the gpu has finished rendering the last frame. Timeout of 1 second
    VK_CHECK(vkWaitForFences(device, 1, &get_current_frame().render_fence, true, 1000000000));

    get_current_frame().deletion_queue.flush();
    get_current_frame().frame_descriptor_allocator.clear_descriptors(device);

    uint32_t swapchain_image_index;

    //request image from the swapchain
    VkResult e = vkAcquireNextImageKHR(device, swapchain, 1000000000, get_current_frame().swapchain_semaphore, nullptr, &swapchain_image_index);
    if (e == VK_ERROR_OUT_OF_DATE_KHR) {
        resize_requested = true;       
        return ;
    }
    
    //start the command buffer recording
    draw_extent.width = std::min(swapchain_extent.width, draw_image.imageExtent.width) * render_scale;
    draw_extent.height = std::min(swapchain_extent.height, draw_image.imageExtent.height) * render_scale;

    VK_CHECK(vkResetFences(device, 1, &get_current_frame().render_fence));
    VK_CHECK(vkResetCommandBuffer(get_current_frame().main_command_buffer, 0));

    VkCommandBuffer cmd = get_current_frame().main_command_buffer;
    
    //begin the command buffer recording. 
    VkCommandBufferBeginInfo cmd_begin_info = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin_info));

    // transition our main draw image into general layout so we can write into it
    // we will overwrite it all so we dont care about what was the older layout
    vkutil::transition_image(cmd, draw_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    draw_background(cmd);

    //transition the draw image and the swapchain image into their correct transfer layouts
    vkutil::transition_image(cmd, draw_image.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkutil::transition_image(cmd, depth_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    
    draw_geometry(cmd);

    vkutil::transition_image(cmd, draw_image.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkutil::transition_image(cmd, swapchain_images[swapchain_image_index], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    
    // execute a copy from the draw image into the swapchain
    vkutil::copy_image_to_image(cmd, draw_image.image, swapchain_images[swapchain_image_index], draw_extent, swapchain_extent);

    // set swapchain image layout to Present so we can show it on the screen
    vkutil::transition_image(cmd, swapchain_images[swapchain_image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    //draw imgui into the swapchain image
    draw_imgui(cmd, swapchain_image_views[swapchain_image_index]);

    //set swapchain image layout to present so we can draw it
    vkutil::transition_image(cmd, swapchain_images[swapchain_image_index], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    
    //finalize the command buffer (we can no longer add commands, but it can now be executed)
    VK_CHECK(vkEndCommandBuffer(cmd));

    //prepare the submission to the queue. 
    //we want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready
    //we will signal the _renderSemaphore, to signal that rendering has finished

    VkCommandBufferSubmitInfo cmd_info = vkinit::command_buffer_submit_info(cmd);

    VkSemaphoreSubmitInfo wait_info = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, get_current_frame().swapchain_semaphore);
    VkSemaphoreSubmitInfo signal_info = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, get_current_frame().render_semaphore);

    VkSubmitInfo2 submit = vkinit::submit_info(&cmd_info, &signal_info, &wait_info);
    
    //submit command buffer to the queue and execute it.
    // render_fence will now block until the graphic commands finish execution
    VK_CHECK(vkQueueSubmit2(graphics_queue, 1, &submit, get_current_frame().render_fence));

    //prepare present
    // this will put the image we just rendered to into the visible window.
    // we want to wait on the render_semaphore for that, 
    // as its necessary that drawing commands have finished before the image is displayed to the user
	VkPresentInfoKHR present_info = vkinit::present_info();
    present_info.pSwapchains = &swapchain;
    present_info.swapchainCount = 1;

    present_info.pWaitSemaphores = &get_current_frame().render_semaphore;
    present_info.waitSemaphoreCount = 1;

    present_info.pImageIndices = &swapchain_image_index;

    VkResult present_result = vkQueuePresentKHR(graphics_queue, &present_info);
    if (present_result == VK_ERROR_OUT_OF_DATE_KHR) {
        resize_requested = true;
    }

    //increase the number of frames drawn
    frame_number++;
}

void VulkanEngine::draw_background(VkCommandBuffer cmd)
{
    ComputeEffect& effect = background_effects[current_background_effect];

    //bind the background compute pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.pipeline);

    //bind the descriptor set containing the draw image for the compute pipeline
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gradient_pipeline_layout, 0, 1, &draw_image_descriptor_set, 0, nullptr);

    vkCmdPushConstants(cmd, gradient_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstants), &effect.constants);
    //execute the compute pipeline dispatch. we are using 16x16 workgroup size so we need to divide by it
    vkCmdDispatch(cmd, std::ceil(draw_extent.width / 16.0f), std::ceil(draw_extent.height / 16.0f), 1);
}

void VulkanEngine::run()
{
    SDL_Event e;
    bool quit = false;

    // main loop
    while (!quit) {
        // Handle events on queue
        while (SDL_PollEvent(&e) != 0) {
            // close the window when user alt-f4s or clicks the X button
            if (e.type == SDL_QUIT) quit = true;
            if (e.type == SDL_WINDOWEVENT) {
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED) {
                    stop_rendering = true;
                }
                if (e.window.event == SDL_WINDOWEVENT_RESTORED) {
                    stop_rendering = false;
                }
            }
            editor_camera.process_sdl_event(e);
            //send sdl event to imgui for handling
            ImGui_ImplSDL2_ProcessEvent(&e);
        }

        // do not draw if we are minimized
        if (stop_rendering) {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        if (resize_requested)
        {
            resize_swapchain();
        }

        //imgui new frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        
        ImGui::NewFrame();

        if (ImGui::Begin("Background"))
        {
            ImGui::SliderFloat("Render Scale",&render_scale, 0.3f, 1.f);

            ComputeEffect& selected = background_effects[current_background_effect];

            ImGui::Text("Selected Effect:", selected.name);
            ImGui::SliderInt("Effect Index", &current_background_effect, 0, background_effects.size() - 1);

            ImGui::InputFloat4("Data 1",reinterpret_cast<float*>(&selected.constants.data1));
            ImGui::InputFloat4("Data 2",reinterpret_cast<float*>(&selected.constants.data2));
            ImGui::InputFloat4("Data 3",reinterpret_cast<float*>(&selected.constants.data3));
            ImGui::InputFloat4("Data 4",reinterpret_cast<float*>(&selected.constants.data4));
        }

        ImGui::End();
        
        //make imgui calculate internal draw structure
        ImGui::Render();

        draw();
    }
}

void VulkanEngine::init_vulkan()
{
    vkb::InstanceBuilder builder;
    vkb::Result<vkb::Instance> inst_ret = builder.set_app_name("Vulkan Engine")
        .request_validation_layers(USE_VALIDATION_LAYERS)
        .use_default_debug_messenger()
        .require_api_version(1, 3, 0)
        .build();
    
    vkb::Instance vkb_inst = inst_ret.value();

    //grab the instance 
    instance = vkb_inst.instance;
    debug_messenger = vkb_inst.debug_messenger;

    SDL_Vulkan_CreateSurface(window, instance, &surface);

    //vulkan 1.3 features
    VkPhysicalDeviceVulkan13Features features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    features.dynamicRendering = VK_TRUE;
    features.synchronization2 = VK_TRUE;

    //vulkan 1.2 features
    VkPhysicalDeviceVulkan12Features features12{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    features12.bufferDeviceAddress = VK_TRUE;
    features12.descriptorIndexing = VK_TRUE;

    //use vkbootstrap to select a gpu. 
    //We want a gpu that can write to the SDL surface and supports vulkan 1.3 with the correct features
    vkb::PhysicalDeviceSelector selector{ vkb_inst };
    vkb::PhysicalDevice physical_device = selector
        .set_minimum_version(1, 3)
        .set_required_features_13(features)
        .set_required_features_12(features12)
        .set_surface(surface)
        .select()
        .value();

    //create the final vulkan device
    vkb::DeviceBuilder device_builder{ physical_device };

    vkb::Device vkb_device = device_builder.build().value();

    // Get the VkDevice handle used in the rest of a vulkan application
    device = vkb_device.device;
    chosen_gpu = physical_device.physical_device;

    graphics_queue = vkb_device.get_queue(vkb::QueueType::graphics).value();
    graphics_queue_family = vkb_device.get_queue_index(vkb::QueueType::graphics).value();

    //Vma Init
    // Initialize memory allocator
    VmaAllocatorCreateInfo allocator_info = {};
    allocator_info.physicalDevice = chosen_gpu;
    allocator_info.device = device;
    allocator_info.instance = instance;
    allocator_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocator_info, &allocator);

    main_deletion_queue.push_function([&]()
    {
        vmaDestroyAllocator(allocator);
    });
}

void VulkanEngine::init_swapchain()
{
    create_swapchain(window_extent.width, window_extent.height);

    // init swap
    // draw image size will math the window
    VkExtent3D draw_image_extent = {
        window_extent.width,
        window_extent.height,
        1
    };

    //hardcoding the draw format to 32 bit float
    draw_image.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    draw_image.imageExtent = draw_image_extent;

    VkImageUsageFlags draw_image_usages{};
    draw_image_usages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    draw_image_usages |= VK_IMAGE_USAGE_STORAGE_BIT;
    draw_image_usages |=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkImageCreateInfo rimg_info = vkinit::image_create_info(draw_image.imageFormat, draw_image_usages, draw_image_extent);

    //for the draw image we want to allocate it from gpu local memory
    VmaAllocationCreateInfo rimg_alloc_info = {};
    rimg_alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    rimg_alloc_info.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    //allocate and create the image
    vmaCreateImage(allocator, &rimg_info, &rimg_alloc_info, &draw_image.image, &draw_image.allocation, nullptr);

    //build a image-view for the draw image to use for rendering
    VkImageViewCreateInfo rview_info = vkinit::imageview_create_info(draw_image.imageFormat, draw_image.image, VK_IMAGE_ASPECT_COLOR_BIT);

    VK_CHECK(vkCreateImageView(device, &rview_info, nullptr, &draw_image.imageView));

    depth_image.imageFormat = VK_FORMAT_D32_SFLOAT;
    depth_image.imageExtent = draw_image_extent;
    VkImageUsageFlags depth_image_usage{};
    depth_image_usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VkImageCreateInfo dimg_info = vkinit::image_create_info(depth_image.imageFormat, depth_image_usage, draw_image_extent);

    //allocate and create the image
    vmaCreateImage(allocator, &dimg_info, &rimg_alloc_info, &depth_image.image, &depth_image.allocation, nullptr);

    //build a image-view for the draw image to use for rendering
    VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(depth_image.imageFormat, depth_image.image, VK_IMAGE_ASPECT_DEPTH_BIT);

    VK_CHECK(vkCreateImageView(device, &dview_info, nullptr, &depth_image.imageView));
    
    // add to deletion queue
    main_deletion_queue.push_function([=]()
    {
        vkDestroyImageView(device, draw_image.imageView, nullptr);
        vmaDestroyImage(allocator, draw_image.image, draw_image.allocation);

        vkDestroyImageView(device, depth_image.imageView, nullptr);
        vmaDestroyImage(allocator, depth_image.image, depth_image.allocation);
    });
}

void VulkanEngine::init_commands()
{
    // create a command pool for the commands submitted to the graphics queue
    // we also want the pool to allow for resetting of individual command buffers
    VkCommandPoolCreateInfo command_pool_info = vkinit::command_pool_create_info(graphics_queue_family, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    for (int i = 0; i < FRAME_OVERLAP; i++)
    {
        VK_CHECK(vkCreateCommandPool(device, &command_pool_info, nullptr, &frames[i].command_pool));

        // allocate the main command buffer from the command pool
		VkCommandBufferAllocateInfo cmd_alloc_info = vkinit::command_buffer_allocate_info(frames[i].command_pool, 1);

        VK_CHECK(vkAllocateCommandBuffers(device, &cmd_alloc_info, &frames[i].main_command_buffer));
    }

    VK_CHECK(vkCreateCommandPool(device, &command_pool_info, nullptr, &immediate_command_pool));

    //allocate the command buffer for immediate submits
    VkCommandBufferAllocateInfo cmd_alloc_info = vkinit::command_buffer_allocate_info(immediate_command_pool, 1);

    VK_CHECK(vkAllocateCommandBuffers(device, &cmd_alloc_info, &immediate_command_buffer));

    main_deletion_queue.push_function([=]()
    {
        vkDestroyCommandPool(device, immediate_command_pool, nullptr);
    });
}

void VulkanEngine::init_sync_structures()
{
    //create syncronization structures
    //one fence to control when the gpu has finished rendering the frame,
    //and 2 semaphores to syncronize rendering with swapchain
    //we want the fence to start signalled so we can wait on it on the first frame
    VkFenceCreateInfo fence_create_info = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphore_create_info = vkinit::semaphore_create_info();

    for (int i = 0; i < FRAME_OVERLAP; i++)
    {
        VK_CHECK(vkCreateFence(device, &fence_create_info, nullptr, &frames[i].render_fence));
        VK_CHECK(vkCreateSemaphore(device, &semaphore_create_info, nullptr, &frames[i].swapchain_semaphore));
        VK_CHECK(vkCreateSemaphore(device, &semaphore_create_info, nullptr, &frames[i].render_semaphore));
    }

    VK_CHECK(vkCreateFence(device, &fence_create_info, nullptr, &immediate_fence));
    main_deletion_queue.push_function([=](){vkDestroyFence(device, immediate_fence, nullptr);});
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height)
{
    vkb::SwapchainBuilder swapchain_builder(chosen_gpu, device, surface);

    swapchain_image_format = VK_FORMAT_B8G8R8A8_UNORM;

    vkb::Swapchain vkb_swapchain = swapchain_builder
    .set_desired_format(VkSurfaceFormatKHR {.format = swapchain_image_format, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
    // use vsync present mode
    .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
    .set_desired_extent(width, height)
    .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
    .build()
    .value();

    swapchain_extent = vkb_swapchain.extent;

    // store swpchain and its images
    swapchain = vkb_swapchain.swapchain;
    swapchain_images = vkb_swapchain.get_images().value();
    swapchain_image_views = vkb_swapchain.get_image_views().value();
}

void VulkanEngine::resize_swapchain()
{
    vkDeviceWaitIdle(device);

    destroy_swapchain();

    int width, height;
    SDL_GetWindowSize(window, &width, &height);

    window_extent.width = width;
    window_extent.height = height;

    create_swapchain(window_extent.width, window_extent.height);

    resize_requested = false;
}

void VulkanEngine::destroy_swapchain()
{
    vkDestroySwapchainKHR(device, swapchain, nullptr);

    // destroy swapchain resource
    for (int i = 0; i < swapchain_images.size(); i++)
    {
        vkDestroyImageView(device, swapchain_image_views[i], nullptr);
    }
}

void VulkanEngine::init_descriptors()
{
    //create a descriptor pool that will hold 10 sets with 1 image each
    std::vector<ExtendableDescriptorAllocator::PoolSizeRatio> sizes = { {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1} };

    global_descriptor_allocator.init_pool(device, 10, sizes);

    //make the descriptor set layout for our compute draw
    {
        DescriptorSetLayout builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        draw_image_descriptor_set_layout = builder.build(device, VK_SHADER_STAGE_COMPUTE_BIT);
    }

    {
        DescriptorSetLayout builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        single_image_descriptor_set_layout = builder.build(device, VK_SHADER_STAGE_FRAGMENT_BIT);
        
    }
    
    // make the descriptor set layout for our mesh rendering
    {
        DescriptorSetLayout builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        gpu_scene_descriptor_set_layout = builder.build(device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    draw_image_descriptor_set = global_descriptor_allocator.allocate(device, draw_image_descriptor_set_layout);

    DescriptorWriter writer;
    writer.write_image(0, draw_image.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    writer.update_set(device, draw_image_descriptor_set);

    //make sure both the descriptor allocator and the new layout get cleaned up properly
    main_deletion_queue.push_function([&]()
    {
       global_descriptor_allocator.destroy_pool(device);
        vkDestroyDescriptorSetLayout(device, draw_image_descriptor_set_layout, nullptr);
    });

    for (int i = 0; i < FRAME_OVERLAP; i++)
    {
        std::vector<ExtendableDescriptorAllocator::PoolSizeRatio> frame_sizes = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4}
        };
        frames[i].frame_descriptor_allocator = ExtendableDescriptorAllocator{};
        frames[i].frame_descriptor_allocator.init_pool(device, 1000, frame_sizes);
        main_deletion_queue.push_function([&, i]()
        {
            frames[i].frame_descriptor_allocator.destroy_pool(device);
        });
    }
}

void VulkanEngine::init_background_pipelines()
{
    VkPipelineLayoutCreateInfo compute_layout{};
    compute_layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    compute_layout.pNext = nullptr;
    compute_layout.pSetLayouts = &draw_image_descriptor_set_layout;
    compute_layout.setLayoutCount = 1;

    VkPushConstantRange push_constant{};
    push_constant.offset = 0;
    push_constant.size = sizeof(ComputePushConstants);
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    compute_layout.pPushConstantRanges = &push_constant;
    compute_layout.pushConstantRangeCount = 1;
    
    VK_CHECK(vkCreatePipelineLayout(device, &compute_layout, nullptr, &gradient_pipeline_layout));

    //layout code
    VkShaderModule gradient_shader;

    if (!vkutil::load_shader_module("d:/Vulkan/vulkan-guide-starting-point-2/shaders/gradient_color.comp.spv", device, &gradient_shader))
    {
        // throw error
        throw std::runtime_error("failed to load shader module: gradient.comp.spv not found");   
    }

    VkShaderModule sky_shader;
    
    if (!vkutil::load_shader_module("d:/Vulkan/vulkan-guide-starting-point-2/shaders/sky.comp.spv", device, &sky_shader))
    {
        // throw error
        throw std::runtime_error("failed to load shader module: gradient.comp.spv not found");   
    }

    VkPipelineShaderStageCreateInfo stage_info{};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.pNext = nullptr;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = gradient_shader;
    stage_info.pName = "main";

    VkComputePipelineCreateInfo compute_pipeline_create_info{};
    compute_pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_pipeline_create_info.pNext = nullptr;
    compute_pipeline_create_info.layout = gradient_pipeline_layout;
    compute_pipeline_create_info.stage = stage_info;

    ComputeEffect gradient;
    gradient.pipeline_layout = gradient_pipeline_layout;
    gradient.name = "Gradient";
    gradient.constants = {};

    //default colors
    gradient.constants.data1 = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
    gradient.constants.data2 = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);

    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compute_pipeline_create_info, nullptr, &gradient.pipeline));

    //change the shader module only to create the sky shader
    compute_pipeline_create_info.stage.module = sky_shader;

    ComputeEffect sky;
    sky.pipeline_layout = gradient_pipeline_layout;
    sky.name = "Sky";
    sky.constants = {};
    
    //default sky parameters
    sky.constants.data1 = glm::vec4(0.1f, 0.2f, 0.4f, 0.97f);

    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compute_pipeline_create_info, nullptr, &sky.pipeline));

    //add the 2 background effects into the array
    background_effects.push_back(gradient);
    background_effects.push_back(sky);

    //destroy the shader modules properly
    vkDestroyShaderModule(device, gradient_shader, nullptr);
    vkDestroyShaderModule(device, sky_shader, nullptr);
    main_deletion_queue.push_function([=]()
    {
        vkDestroyPipelineLayout(device, gradient_pipeline_layout, nullptr);
        vkDestroyPipeline(device, sky.pipeline, nullptr);
        vkDestroyPipeline(device, gradient.pipeline, nullptr);
    });
}

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& func)
{
    VK_CHECK(vkResetFences(device, 1, &immediate_fence));
    VK_CHECK(vkResetCommandBuffer(immediate_command_buffer, 0));

    VkCommandBuffer cmd = immediate_command_buffer;

    VkCommandBufferBeginInfo cmd_begin_info = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin_info));

	func(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmd_info = vkinit::command_buffer_submit_info(cmd);
    VkSubmitInfo2 submit = vkinit::submit_info(&cmd_info, nullptr, nullptr);
    
    //submit command buffer to the queue and execute it
    // render_fence will now block until the graphic commands finish execution
    VK_CHECK(vkQueueSubmit2(graphics_queue, 1, &submit, immediate_fence));

    VK_CHECK(vkWaitForFences(device, 1, &immediate_fence, true, 9999999999));
}

void VulkanEngine::draw_imgui(VkCommandBuffer cmd, VkImageView target_image_view)
{
    VkRenderingAttachmentInfo color_attachment = vkinit::attachment_info(target_image_view, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingInfo render_info = vkinit::rendering_info(swapchain_extent, &color_attachment, nullptr);

    vkCmdBeginRendering(cmd, &render_info);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    vkCmdEndRendering(cmd);
}

void VulkanEngine::draw_geometry(VkCommandBuffer cmd)
{
    //begin a render pass connected to our draw image
    VkRenderingAttachmentInfo color_attachment = vkinit::attachment_info(draw_image.imageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo depth_attachment = vkinit::depth_attachment_info(depth_image.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    
    VkRenderingInfo render_info = vkinit::rendering_info(draw_extent, &color_attachment, &depth_attachment);
    vkCmdBeginRendering(cmd, &render_info);

    //set dynamic viewport and scissor
    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = draw_extent.width;
    viewport.height = draw_extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    
    VkRect2D scissor = {};
    scissor.offset.x = 0.0f;
    scissor.offset.y = 0.0f;
    scissor.extent.width = draw_extent.width;
    scissor.extent.height = draw_extent.height;
    
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // allocate a new uniform buffer for the scene data
    AllocatedBuffer gpu_scene_data_buffer = create_buffer(sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT , VMA_MEMORY_USAGE_CPU_TO_GPU);

    // add it to the deletion queue of this frame so it gets deleted once its been used
    get_current_frame().deletion_queue.push_function([=, this]()
    {
        destroy_buffer(gpu_scene_data_buffer);
    });

    // write the buffer
    GPUSceneData* scene_uniform_data = static_cast<GPUSceneData*>(gpu_scene_data_buffer.allocation->GetMappedData());
    *scene_uniform_data = scene_data;

    // create a descriptor set that binds that buffer and update it
    VkDescriptorSet global_descriptor = get_current_frame().frame_descriptor_allocator.allocate(device, gpu_scene_descriptor_set_layout);

    DescriptorWriter writer;

    writer.write_buffer(0, gpu_scene_data_buffer.buffer, sizeof(GPUSceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.update_set(device, global_descriptor);

    for (const RenderObject& draw : main_draw_context.opaque_surfaces)
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipeline->pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipeline->layout, 0, 1, &global_descriptor, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipeline->layout, 1, 1, &draw.material->material_set, 0, nullptr);

        vkCmdBindIndexBuffer(cmd, draw.index_buffer, 0, VK_INDEX_TYPE_UINT32);

        GPUDrawPushConstants push_constants;
        push_constants.vertexBuffer = draw.vertex_buffer_address;
        push_constants.worldMatrix = draw.transform;
        vkCmdPushConstants(cmd, draw.material->pipeline->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants), &push_constants);
        vkCmdDrawIndexed(cmd, draw.index_count, 1, draw.first_index, 0, 0);
    }
    
    vkCmdEndRendering(cmd);
}

AllocatedBuffer VulkanEngine::create_buffer(size_t size, VkBufferUsageFlags usage, VmaMemoryUsage memory_usage)
{
    //allocate buffer
    VkBufferCreateInfo buffer_create_info = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_create_info.pNext = nullptr;
    buffer_create_info.size = size;
    buffer_create_info.usage = usage;

    VmaAllocationCreateInfo allocation_create_info = {};
    allocation_create_info.usage = memory_usage;
    allocation_create_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    AllocatedBuffer new_buffer;

    //allocate the buffer
    VK_CHECK(vmaCreateBuffer(allocator, &buffer_create_info, &allocation_create_info, &new_buffer.buffer, &new_buffer.allocation, &new_buffer.allocationInfo));

    return new_buffer;
}

void VulkanEngine::destroy_buffer(const AllocatedBuffer& buffer)
{
    vmaDestroyBuffer(allocator, buffer.buffer, buffer.allocation);
}

GPUMeshBuffers VulkanEngine::upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices)
{
    const size_t vertex_buffer_size = vertices.size() * sizeof(Vertex);
    const size_t index_buffer_size = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers new_surface;

    new_surface.vertexBuffer = create_buffer(vertex_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    // find the address of the vertex buffer
    VkBufferDeviceAddressInfo device_address_info {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = new_surface.vertexBuffer.buffer};
    new_surface.vertexBufferAddress = vkGetBufferDeviceAddress(device, &device_address_info);

    //create index buffer
    new_surface.indexBuffer = create_buffer(index_buffer_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    AllocatedBuffer staging_buffer = create_buffer(vertex_buffer_size + index_buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    void* data = staging_buffer.allocation->GetMappedData();

    //copy vertex buffer
    memcpy(data, vertices.data(), vertex_buffer_size);
    //copy index buffer
    memcpy(static_cast<char*>(data) + vertex_buffer_size, indices.data(), index_buffer_size);
    
	immediate_submit([&](VkCommandBuffer cmd)
    {
        VkBufferCopy vertex_copy {0};
        vertex_copy.dstOffset = 0;
        vertex_copy.srcOffset = 0;
        vertex_copy.size = vertex_buffer_size;

        vkCmdCopyBuffer(cmd, staging_buffer.buffer, new_surface.vertexBuffer.buffer, 1, &vertex_copy);

        VkBufferCopy index_copy {0};
        index_copy.dstOffset = 0;
        index_copy.srcOffset = vertex_buffer_size;
        index_copy.size = index_buffer_size;

        vkCmdCopyBuffer(cmd, staging_buffer.buffer, new_surface.indexBuffer.buffer, 1, &index_copy);
    });

    destroy_buffer(staging_buffer);

    return new_surface;
}

void VulkanEngine::init_mesh_pipeline()
{
    VkShaderModule triangle_frag_shader;
    if (!vkutil::load_shader_module("d:/Vulkan/vulkan-guide-starting-point-2/shaders/tex_image.frag.spv", device, &triangle_frag_shader))
    {
        // throw error
        throw std::runtime_error("failed to load shader module: tex_image.frag.spv not found");   
    }
    
    fmt::println("Triangle fragment shader successfully loaded");

    VkShaderModule triangle_vert_shader;
    if (!vkutil::load_shader_module("d:/Vulkan/vulkan-guide-starting-point-2/shaders/colored_triangle_mesh.vert.spv", device, &triangle_vert_shader))
    {
        // throw error
        throw std::runtime_error("failed to load shader module: colored_triangle_mesh.vert.spv not found");   
    }

    fmt::println("Triangle vertex shader successfully loaded");

    VkPushConstantRange buffer_range{};
    buffer_range.offset = 0;
    buffer_range.size = sizeof(GPUDrawPushConstants);
    buffer_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
    pipeline_layout_info.pPushConstantRanges = &buffer_range;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pSetLayouts = &single_image_descriptor_set_layout;
    pipeline_layout_info.setLayoutCount = 1;
    
    VK_CHECK(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &mesh_pipeline_layout));

    PipelineBuilder pipeline_builder;

    //use the triangle layout we created
    pipeline_builder.pipeline_layout = mesh_pipeline_layout;
    //connect the vertex and pixel shader to the pipeline
    pipeline_builder.set_shaders(triangle_vert_shader, triangle_frag_shader);
    //it will draw triangles
    pipeline_builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    //filled triangles
    pipeline_builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    //no backface culling
    pipeline_builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    //no multisampling
    pipeline_builder.set_multisampling_none();
    //disable blending
    // pipeline_builder.disable_blending();

    pipeline_builder.enable_blending_additive();
    //disable depth test
    // pipeline_builder.disable_depthtest();
    pipeline_builder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);
    
    //connect the image format we will draw into, from draw image
    pipeline_builder.set_color_attachment_format(draw_image.imageFormat);
    //no depth buffer
    pipeline_builder.set_depth_format(depth_image.imageFormat);

    //finally build the pipeline
    mesh_pipeline = pipeline_builder.build_pipeline(device);
    //clean structures
    vkDestroyShaderModule(device, triangle_frag_shader, nullptr);
    vkDestroyShaderModule(device, triangle_vert_shader, nullptr);
    main_deletion_queue.push_function([&]() {
        vkDestroyPipelineLayout(device, mesh_pipeline_layout, nullptr);
        vkDestroyPipeline(device, mesh_pipeline, nullptr);
    });
}

AllocatedImage VulkanEngine::create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped)
{
    AllocatedImage new_image;
    new_image.imageFormat = format;
    new_image.imageExtent = size;

    VkImageCreateInfo image_create_info = vkinit::image_create_info(format, usage, size);
    if (mipmapped)
    {
        image_create_info.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(size.width, size.height)))) + 1;
    }

    // always allocate images on dedicated gpu memory
    VmaAllocationCreateInfo allocation_info = {.usage = VMA_MEMORY_USAGE_GPU_ONLY};
    allocation_info.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // allocate and create the image
    VK_CHECK(vmaCreateImage(allocator, &image_create_info, &allocation_info, &new_image.image, &new_image.allocation, nullptr));

    // if the format is a depth format, we will need to have it use the correct aspect flag
    VkImageAspectFlags aspect_flags = VK_IMAGE_ASPECT_COLOR_BIT;
    if (format == VK_FORMAT_D32_SFLOAT)
    {
        aspect_flags = VK_IMAGE_ASPECT_DEPTH_BIT;
    }

    // build a image-view for the image
    VkImageViewCreateInfo view_info = vkinit::imageview_create_info(format, new_image.image, aspect_flags);
    view_info.subresourceRange.levelCount = image_create_info.mipLevels;

    VK_CHECK(vkCreateImageView(device, &view_info, nullptr, &new_image.imageView));

    return new_image;
}

AllocatedImage VulkanEngine::create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage,
    bool mipmapped)
{
    size_t data_size = size.depth * size.width * size.height * 4;
    AllocatedBuffer upload_buffer = create_buffer(data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    memcpy(upload_buffer.allocationInfo.pMappedData, data, data_size);

    AllocatedImage new_image = create_image(size, format, usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, mipmapped);

    immediate_submit([&](VkCommandBuffer cmd)
    {
        vkutil::transition_image(cmd, new_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkBufferImageCopy image_copy = {};
        image_copy.bufferOffset = 0;
        image_copy.bufferRowLength = 0;
        image_copy.bufferImageHeight = 0;
        
        image_copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        image_copy.imageSubresource.mipLevel = 0;
        image_copy.imageSubresource.baseArrayLayer = 0;
        image_copy.imageSubresource.layerCount = 1;
        image_copy.imageExtent = size;

        // copy the buffer into the image
        vkCmdCopyBufferToImage(cmd, upload_buffer.buffer, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &image_copy);

        vkutil::transition_image(cmd, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });

    destroy_buffer(upload_buffer);
    return new_image;
}

void VulkanEngine::destroy_image(const AllocatedImage& image)
{
    vkDestroyImageView(device, image.imageView, nullptr);
    vmaDestroyImage(allocator, image.image, image.allocation);
}

void VulkanEngine::init_imgui()
{
    // create descriptor pool for imgui
    // the size of the pool is very oversize, but it's copied from imgui demo itself
    VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
    };

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000;
    pool_info.poolSizeCount = static_cast<uint32_t>(std::size(pool_sizes));
    pool_info.pPoolSizes = pool_sizes;

    VkDescriptorPool imgui_pool;
    VK_CHECK(vkCreateDescriptorPool(device, &pool_info, nullptr, &imgui_pool));

    // initializes imgui library

    //this initializes the core structures of imgui
    ImGui::CreateContext();

    //this initalizes imgui for SDL
    ImGui_ImplSDL2_InitForVulkan(window);

    //this initializes imgui for vulkan
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = instance;
    init_info.PhysicalDevice = chosen_gpu;
    init_info.Device = device;
    init_info.Queue = graphics_queue;
    init_info.DescriptorPool = imgui_pool;
    init_info.MinImageCount = 3;
    init_info.ImageCount = 3;
    init_info.UseDynamicRendering = true;

    //dynamic rendering parameters for imgui to use
    init_info.PipelineRenderingCreateInfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    init_info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    init_info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &swapchain_image_format;

    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&init_info);

    ImGui_ImplVulkan_CreateFontsTexture();

    // add the destroy to imgui created structures
    main_deletion_queue.push_function([=]()
    {
        ImGui_ImplVulkan_Shutdown();
        vkDestroyDescriptorPool(device, imgui_pool, nullptr);
    });
}

void VulkanEngine::init_default_data()
{
    std::array<Vertex,4> rect_vertices;

    rect_vertices[0].position = {0.5,-0.5, 0};
    rect_vertices[1].position = {0.5,0.5, 0};
    rect_vertices[2].position = {-0.5,-0.5, 0};
    rect_vertices[3].position = {-0.5,0.5, 0};

    rect_vertices[0].color = {0,0, 0,1};
    rect_vertices[1].color = { 0.5,0.5,0.5 ,1};
    rect_vertices[2].color = { 1,0, 0,1 };
    rect_vertices[3].color = { 0,1, 0,1 };

    std::array<uint32_t,6> rect_indices;

    rect_indices[0] = 0;
    rect_indices[1] = 1;
    rect_indices[2] = 2;

    rect_indices[3] = 2;
    rect_indices[4] = 1;
    rect_indices[5] = 3;

    rectangle = upload_mesh(rect_indices,rect_vertices);

    //delete the rectangle data on engine shutdown
    main_deletion_queue.push_function([&](){
        destroy_buffer(rectangle.indexBuffer);
        destroy_buffer(rectangle.vertexBuffer);
    });

    uint32_t white = glm::packUnorm4x8(glm::vec4(1,1,1,1));
    white_image = create_image((void*)&white, VkExtent3D{1,1,1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);
    
    uint32_t black = glm::packUnorm4x8(glm::vec4(0,0,0,0));
    black_image = create_image((void*)&black, VkExtent3D{1,1,1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);
    
    uint32_t grey = glm::packUnorm4x8(glm::vec4(0.66f, 0.66f, 0.66f, 1));
    grey_image = create_image((void*)&grey, VkExtent3D{1,1,1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    //Checkerboard image
    uint32_t magenta = glm::packUnorm4x8(glm::vec4(1,0,1,1));
    std::array<uint32_t, 16 *16>pixels; //for 16x16 checkerboard texture
    for (int x = 0; x < 16; x++)
    {
        for (int y = 0; y < 16; y++)
        {
            pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
        }
    }
    error_checkerboard_image = create_image(pixels.data(), VkExtent3D{16, 16, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    VkSamplerCreateInfo sampler = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};

    sampler.magFilter = VK_FILTER_NEAREST;
    sampler.minFilter = VK_FILTER_NEAREST;

    vkCreateSampler(device, &sampler, nullptr, &default_sampler_nearest);

    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
    vkCreateSampler(device, &sampler, nullptr, &default_sampler_linear);

    main_deletion_queue.push_function([&]()
    {
        vkDestroySampler(device, default_sampler_nearest, nullptr);
        vkDestroySampler(device, default_sampler_linear, nullptr);
        destroy_image(white_image);
        destroy_image(black_image);
        destroy_image(grey_image);
        destroy_image(error_checkerboard_image);
    });

    GLTF_metallic_roughness::MaterialResource material_resource;
    // default the material textures
    material_resource.color_image = white_image;
    material_resource.color_sampler = default_sampler_linear;
    material_resource.metal_rough_image = white_image;
    material_resource.metal_rough_sampler = default_sampler_linear;

    // set the uniform buffer for the material data
    AllocatedBuffer material_constants = create_buffer(sizeof(GLTF_metallic_roughness::MaterialConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    // write the buffer
    GLTF_metallic_roughness::MaterialConstants* scene_uniform_data = static_cast<GLTF_metallic_roughness::MaterialConstants*>(material_constants.allocation->GetMappedData());
    scene_uniform_data->baseColorFactor = glm::vec4(1,1,1,1);
    scene_uniform_data->metallicRoughnessFactor = glm::vec4(1.0f, 0.5f, 0, 0);

    main_deletion_queue.push_function([=,this]()
    {
        destroy_buffer(material_constants);
    });

    material_resource.data_buffer = material_constants.buffer;
    material_resource.data_buffer_offset = 0;

    default_data = metallic_roughness.write_material(device, MaterialPass::MainColor, material_resource, global_descriptor_allocator);
    
    meshes = load_gltf_meshes(this, "D:/Vulkan/vulkan-guide-starting-point-2/assets/basicmesh.glb").value();
    if (meshes.size() == 0)
    {
        throw std::runtime_error("failed to load meshes");   
    }
    fmt::println("Mesh loaded, {} meshes in file", meshes.size());

    for (std::shared_ptr<MeshAsset>& mesh : meshes)
    {
        std::shared_ptr<MeshNode> new_node = std::make_shared<MeshNode>();
        new_node->mesh = mesh;
        new_node->local_transform = glm::mat4(1);
        new_node->world_transform = glm::mat4(1);
        for(GeoSurface& surface : new_node->mesh->surfaces)
        {
            surface.material = std::make_shared<GLTFMaterial>(default_data);
        }
        loaded_nodes[mesh->name] = std::move(new_node);
    }
}

void GLTF_metallic_roughness::build_pipelines(VulkanEngine* engine)
{
    VkShaderModule mesh_frag_shader, mesh_vert_shader;
    if (!vkutil::load_shader_module("D:/Vulkan/vulkan-guide-starting-point-2/shaders/mesh.frag.spv", engine->device, &mesh_frag_shader))
    {
        fmt::println("failed to load shader module: mesh.frag.spv not found");
    }
    if (!vkutil::load_shader_module("D:/Vulkan/vulkan-guide-starting-point-2/shaders/mesh.vert.spv", engine->device, &mesh_vert_shader))
    {
        fmt::println("failed to load shader module: mesh.vert.spv not found");
    }
    VkPushConstantRange matrix_range {};
    matrix_range.offset = 0;
    matrix_range.size = sizeof(GPUDrawPushConstants);
    matrix_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    DescriptorSetLayout layout_builder;
    layout_builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    layout_builder.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    layout_builder.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    material_layout = layout_builder.build(engine->device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

    VkDescriptorSetLayout layouts[] = { engine->gpu_scene_descriptor_set_layout, material_layout};

    VkPipelineLayoutCreateInfo mesh_layout_info = vkinit::pipeline_layout_create_info();
    mesh_layout_info.setLayoutCount = 2;
    mesh_layout_info.pSetLayouts = layouts;
    mesh_layout_info.pPushConstantRanges = &matrix_range;
    mesh_layout_info.pushConstantRangeCount = 1;
    
    VkPipelineLayout new_layout;

    VK_CHECK(vkCreatePipelineLayout(engine->device, &mesh_layout_info, nullptr, &new_layout));

    opaque_pipeline.layout = new_layout;
    transparent_pipeline.layout = new_layout;

    // build the stage create info for both vertex and fragment stage, this let the pipeline know the shader modules per stage
    PipelineBuilder pipeline_builder;
    pipeline_builder.set_shaders(mesh_vert_shader, mesh_frag_shader);
    pipeline_builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipeline_builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipeline_builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    pipeline_builder.set_multisampling_none();
    pipeline_builder.disable_blending();
    pipeline_builder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);

    // render format
    pipeline_builder.set_color_attachment_format(engine->draw_image.imageFormat);
    pipeline_builder.set_depth_format(engine->depth_image.imageFormat);

    // use the triangle layout we created
    pipeline_builder.pipeline_layout = new_layout;

    // finally build the pipeline
    opaque_pipeline.pipeline = pipeline_builder.build_pipeline(engine->device);

    // create the transparent variant
    pipeline_builder.enable_blending_additive();

    pipeline_builder.enable_depthtest(false, VK_COMPARE_OP_GREATER_OR_EQUAL);

    vkDestroyShaderModule(engine->device, mesh_frag_shader, nullptr);
    vkDestroyShaderModule(engine->device, mesh_vert_shader, nullptr);
}

MaterialInstance GLTF_metallic_roughness::write_material(VkDevice device, MaterialPass pass, const MaterialResource& resource, ExtendableDescriptorAllocator descriptor_allocator)
{
    MaterialInstance material_instance;
    material_instance.pass_type = pass;
    if (pass == MaterialPass::Transparent)
    {
        material_instance.pipeline = &transparent_pipeline;
    }
    else
    {
        material_instance.pipeline = &opaque_pipeline;  
    }
    material_instance.material_set = descriptor_allocator.allocate(device, material_layout);

    writer.clear();
    writer.write_buffer(0, resource.data_buffer, sizeof(MaterialConstants), resource.data_buffer_offset, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.write_image(1, resource.color_image.imageView, resource.color_sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(2,resource.metal_rough_image.imageView, resource.metal_rough_sampler,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    writer.update_set(device, material_instance.material_set);

    return material_instance;
}

void MeshNode::Draw(const glm::mat4& top_matrix, DrawContext& ctx)
{
    glm::mat4 node_matrix = top_matrix * world_transform;

    for (auto& surface : mesh->surfaces)
    {
        RenderObject object;
        object.index_count = surface.index_count;
        object.first_index = surface.start_index;
        object.index_buffer = mesh->mesh_buffers.indexBuffer.buffer;
        object.material = &surface.material->data;

        object.transform = node_matrix;
        object.vertex_buffer_address = mesh->mesh_buffers.vertexBufferAddress;
        ctx.opaque_surfaces.push_back(object);
    }
    Node::Draw(top_matrix, ctx);   
}


void VulkanEngine::update_scene()
{

    //
    // for (std::pair<const std::string, std::shared_ptr<Node>>& node : loaded_nodes) {
    //     node.second->Draw(glm::mat4{1.f}, main_draw_context);
    // }

    for (int x = -3; x < 3; x++) {

        glm::mat4 scale = glm::scale(glm::vec3{0.2});
        glm::mat4 translation =  glm::translate(glm::vec3{x, 1, 0});

        loaded_nodes["Cube"]->Draw(translation * scale, main_draw_context);
    }

    
    editor_camera.update();

    glm::mat4 view = editor_camera.get_view_matrix();

    // camera projection
    glm::mat4 projection = glm::perspective(glm::radians(70.f), static_cast<float>(window_extent.width) / static_cast<float>(window_extent.height), 1000.0f, 0.1f);

    // invert the y direction on projection matrix so that we are more similar to opengl and gltf axis
    projection[1][1] *= -1;
    scene_data.view = view;
    scene_data.projection = projection;
    scene_data.view_projection = projection * view;

    
    //some default lighting params
    scene_data.ambient_color = glm::vec4(0.1f);
    scene_data.sunlight_color = glm::vec4(1.0f);
    scene_data.sunlight_direction = glm::vec4(0.0f, 1.0f, 0.5f, 1.0f);
}

void GLTF_metallic_roughness::clear_resources(VkDevice device)
{
    vkDestroyDescriptorSetLayout(device,material_layout,nullptr);
    vkDestroyPipelineLayout(device,transparent_pipeline.layout,nullptr);

    vkDestroyPipeline(device, transparent_pipeline.pipeline, nullptr);
    vkDestroyPipeline(device, opaque_pipeline.pipeline, nullptr);
}
