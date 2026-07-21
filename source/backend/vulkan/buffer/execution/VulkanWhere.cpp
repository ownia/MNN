//
//  VulkanWhere.cpp
//  MNN
//

#include "VulkanWhere.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {

struct WhereParam {
    ivec4 size;
    ivec4 strides0;
    ivec4 strides1;
};

VulkanWhere::VulkanWhere(Tensor* input, Backend* backend) : VulkanBasicExecution(backend) {
    auto vkBackend = static_cast<VulkanBackend*>(backend);
    mParam = vkBackend->allocUniform(nullptr, sizeof(WhereParam));
    std::vector<VkDescriptorType> types{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };

    std::string shaderName = "glsl_where_";
    if (input->getType().code == halide_type_int) {
        shaderName += "USE_INT_";
    } else if (vkBackend->useFP16()) {
        shaderName += "FP16_";
    }
    shaderName += "comp";
    mPipeline = vkBackend->getPipeline(shaderName, types);
    mDescriptorSet.reset(mPipeline->createSet());
}

VulkanWhere::~VulkanWhere() {
    auto vkBackend = static_cast<VulkanBackend*>(backend());
    vkBackend->recycleUniform(mParam);
}

ErrorCode VulkanWhere::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input = inputs[0];
    auto output = outputs[0];
    auto param = reinterpret_cast<WhereParam*>(mParam->map());
    param->size[0] = input->elementSize();
    param->size[1] = input->dimensions();
    param->size[2] = output->length(0);
    param->size[3] = 0;
    for (int i = 0; i < 4; ++i) {
        param->strides0[i] = i < input->dimensions() ? input->stride(i) : 0;
        param->strides1[i] = i + 4 < input->dimensions() ? input->stride(i + 4) : 0;
    }
    mParam->unmap();

    auto vkBackend = static_cast<VulkanBackend*>(backend());
    mDescriptorSet->writeBuffer(vkBackend->getBuffer(output), 0);
    mDescriptorSet->writeBuffer(vkBackend->getBuffer(input), 1);
    mDescriptorSet->writeBuffer(mParam->buffer(), 2, mParam->size());
    mPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
    vkCmdDispatch(cmdBuffer->get(), 1, 1, 1);
    return NO_ERROR;
}

class VulkanWhereCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs,
                                            const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                            Backend* backend) const override {
        auto input = inputs[0];
        auto type = input->getType();
        if (TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 ||
            input->dimensions() > 8 ||
            (type.code != halide_type_float && !(type.code == halide_type_int && type.bits == 32))) {
            return nullptr;
        }
        return new VulkanWhere(input, backend);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Where, new VulkanWhereCreator);
    return true;
}();

} // namespace MNN