
#pragma once 
#include <vulkan/vulkan.h>

namespace vkutil {

    void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout);
    void copy_image_to_image(VkCommandBuffer cmd, VkImage src_img, VkImage dst_img, VkExtent2D src_size, VkExtent2D dst_size);
};