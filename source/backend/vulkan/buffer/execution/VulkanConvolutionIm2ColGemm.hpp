#ifndef VulkanConvolutionIm2ColGemm_hpp
#define VulkanConvolutionIm2ColGemm_hpp

#include "VulkanConvolution.hpp"

namespace MNN {

class VulkanConvolutionIm2ColGemm : public VulkanConvolutionCommon {
public:
    VulkanConvolutionIm2ColGemm(VulkanBackend* backend, const Convolution2DCommon* common, const float* weightPtr,
                                const float* biasPtr, int ci, int co);
    virtual bool onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) override;
    bool valid() const;

protected:
    virtual ErrorCode onEncodeConvolution(const Convolution2DCommon* common, const std::vector<Tensor*>& inputs,
                                          const std::vector<Tensor*>& outputs,
                                          const VulkanCommandPool::Buffer* cmdBuffer,
                                          const VulkanBuffer* constConvBuffer) override;

private:
    bool init(VulkanBackend* backend, const Convolution2DCommon* common, const float* weightPtr,
              const float* biasPtr);

    int mCi = 0;
    int mCo = 0;
    int mKernelWidth = 0;
    uint32_t mPadK = 0;
    uint32_t mPadN = 0;
    bool mUseFP16 = false;
    const VulkanPipeline* mPackPipeline = nullptr;
    const VulkanPipeline* mGemmPipeline = nullptr;
    std::shared_ptr<VulkanLayout::DescriptorSet> mPackSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mGemmSet;
    std::shared_ptr<VulkanBuffer> mPackedWeight;
    std::shared_ptr<VulkanBuffer> mBias;
    std::shared_ptr<Tensor> mPackedInput;
};

} // namespace MNN

#endif
