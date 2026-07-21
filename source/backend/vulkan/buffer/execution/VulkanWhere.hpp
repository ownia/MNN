//
//  VulkanWhere.hpp
//  MNN
//

#ifndef VulkanWhere_hpp
#define VulkanWhere_hpp

#include "VulkanBasicExecution.hpp"

namespace MNN {

class VulkanWhere : public VulkanBasicExecution {
public:
    VulkanWhere(Tensor* input, Backend* backend);
    virtual ~VulkanWhere();

    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanBuffer> mParam;
    const VulkanPipeline* mPipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
};

} // namespace MNN

#endif /* VulkanWhere_hpp */