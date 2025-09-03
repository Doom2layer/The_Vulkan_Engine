#include <vk_descriptors.h>

void DescriptorSetLayout::add_binding(uint32_t binding, VkDescriptorType type)
{
    VkDescriptorSetLayoutBinding newbind {};
    newbind.binding = binding;
    newbind.descriptorCount = 1;
    newbind.descriptorType = type;

    bindings.push_back(newbind);
}

void DescriptorSetLayout::clear()
{
    bindings.clear();
}

VkDescriptorSetLayout DescriptorSetLayout::build(VkDevice device, VkShaderStageFlags shaderStages, void* pNext, VkDescriptorSetLayoutCreateFlags flags)
{
    
    for (VkDescriptorSetLayoutBinding& bind : bindings)
    {
        bind.stageFlags = shaderStages;
    }

    VkDescriptorSetLayoutCreateInfo info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.pNext = pNext;
    info.pBindings = bindings.data();
    info.bindingCount = static_cast<uint32_t>(bindings.size());
    info.flags = flags;

    VkDescriptorSetLayout set;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &info, nullptr, &set));

    return set;
}

void DescriptorAllocator::init_pool(VkDevice device, uint32_t max_sets, std::span<PoolSizeRatio> pool_ratios)
{
    std::vector<VkDescriptorPoolSize> pool_sizes;
    for (PoolSizeRatio ratio : pool_ratios)
    {
        pool_sizes.push_back(VkDescriptorPoolSize{.type = ratio.type, .descriptorCount = static_cast<uint32_t>(ratio.ratio * max_sets)});
    }

    VkDescriptorPoolCreateInfo pool_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.flags = 0;
    pool_info.maxSets = max_sets;
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();

    vkCreateDescriptorPool(device, &pool_info, nullptr, &pool);
}

void DescriptorAllocator::clear_descriptors(VkDevice device)
{
    vkResetDescriptorPool(device, pool, 0);
}

void DescriptorAllocator::destroy_pool(VkDevice device)
{
    vkDestroyDescriptorPool(device, pool, nullptr);
}

VkDescriptorSet DescriptorAllocator::allocate(VkDevice device, VkDescriptorSetLayout layout)
{
    VkDescriptorSetAllocateInfo alloc_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc_info.pNext = nullptr;
    alloc_info.descriptorPool = pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &layout;

    VkDescriptorSet set;
    VK_CHECK(vkAllocateDescriptorSets(device, &alloc_info, &set));

    return set;
}

VkDescriptorPool ExtendableDescriptorAllocator::get_pool(VkDevice device)
{
    VkDescriptorPool new_pool;
    if (ready_pools.size() != 0)
    {
        new_pool = ready_pools.back();
        ready_pools.pop_back();
    }
    else
    {
        new_pool = create_pool(device, sets_per_pool, ratios);

        sets_per_pool = sets_per_pool * 1.5;
        if (sets_per_pool > 4092)
        {
            sets_per_pool = 4092;
        }
    }
    return new_pool;
}

VkDescriptorPool ExtendableDescriptorAllocator::create_pool(VkDevice device, uint32_t set_count, std::span<PoolSizeRatio> pool_ratios)
{
    std::vector<VkDescriptorPoolSize> pool_sizes;
    for (PoolSizeRatio ratio : pool_ratios)
    {
        pool_sizes.push_back(VkDescriptorPoolSize{.type = ratio.type, .descriptorCount = static_cast<uint32_t>(ratio.ratio * set_count)});
    }
    VkDescriptorPoolCreateInfo pool_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.flags = 0;
    pool_info.maxSets = set_count;
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();

    VkDescriptorPool new_pool;
    vkCreateDescriptorPool(device, &pool_info, nullptr, &new_pool);
    return new_pool;
}

void ExtendableDescriptorAllocator::init_pool(VkDevice device, uint32_t initial_max_sets, std::span<PoolSizeRatio> pool_ratios)
{
    ratios.clear();

    for (PoolSizeRatio ratio : pool_ratios)
    {
        ratios.push_back(ratio);
    }

    VkDescriptorPool new_pool = create_pool(device, initial_max_sets, pool_ratios);

    sets_per_pool = initial_max_sets * 1.5;

    ready_pools.push_back(new_pool);
}

void ExtendableDescriptorAllocator::clear_descriptors(VkDevice device)
{

    for (VkDescriptorPool pool : ready_pools)
    {
        vkResetDescriptorPool(device, pool, 0);
    }

    for (VkDescriptorPool pool : full_pools)
    {
        vkResetDescriptorPool(device, pool, 0);
        ready_pools.push_back(pool);
    }
    full_pools.clear();
}

void ExtendableDescriptorAllocator::destroy_pool(VkDevice device)
{
    for (VkDescriptorPool pool : ready_pools)
    {
        vkDestroyDescriptorPool(device, pool, nullptr);
    }
    ready_pools.clear();

    for (VkDescriptorPool pool : full_pools)
    {
        vkDestroyDescriptorPool(device, pool, nullptr);
    }
    full_pools.clear();
}

VkDescriptorSet ExtendableDescriptorAllocator::allocate(VkDevice device, VkDescriptorSetLayout layout, void* pNext)
{
    // get or create a pool to allocate from
    VkDescriptorPool pool_to_use = get_pool(device);

    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.pNext = pNext;
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = pool_to_use;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &layout;

    VkDescriptorSet discriptor_set;

    VkResult result = vkAllocateDescriptorSets(device, &alloc_info, &discriptor_set);

    //allocation failed. try again
    if (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL)
    {
        full_pools.push_back(pool_to_use);
        pool_to_use = get_pool(device);
        alloc_info.descriptorPool = pool_to_use;

        VK_CHECK(vkAllocateDescriptorSets(device, &alloc_info, &discriptor_set));
    }
    ready_pools.push_back(pool_to_use);
    return discriptor_set;
}

void DescriptorWriter::write_buffer(int binding, VkBuffer buffer, size_t size, size_t offset, VkDescriptorType type)
{
    VkDescriptorBufferInfo& buffer_info = buffer_infos.emplace_back(VkDescriptorBufferInfo{.buffer = buffer, .offset = offset, .range = size});
    VkWriteDescriptorSet write_info = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write_info.dstBinding = binding;
    write_info.dstSet = VK_NULL_HANDLE;
    write_info.descriptorCount = 1;
    write_info.descriptorType = type;
    write_info.pBufferInfo = &buffer_info;

    writes.push_back(write_info);
}

void DescriptorWriter::write_image(int binding, VkImageView image_view, VkSampler sampler, VkImageLayout image_layout, VkDescriptorType type)
{
    VkDescriptorImageInfo& image_info = image_infos.emplace_back(VkDescriptorImageInfo{.sampler = sampler, .imageView = image_view, .imageLayout = image_layout});
    VkWriteDescriptorSet write_info = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write_info.dstBinding = binding;
    write_info.dstSet = VK_NULL_HANDLE;
    write_info.descriptorCount = 1;
    write_info.descriptorType = type;
    write_info.pImageInfo = &image_info;

    writes.push_back(write_info);
}

void DescriptorWriter::clear()
{
    image_infos.clear();
    writes.clear();
    buffer_infos.clear();
}

void DescriptorWriter::update_set(VkDevice device, VkDescriptorSet set)
{
    for (VkWriteDescriptorSet& write : writes)
    {
        write.dstSet = set;
    }
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

