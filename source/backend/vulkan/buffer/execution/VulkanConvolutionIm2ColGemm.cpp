#include "VulkanConvolutionIm2ColGemm.hpp"
#include "VulkanBackend.hpp"

#include <cstring>
#include <vector>

#include "core/Macro.h"

namespace MNN {

VulkanConvolutionIm2ColGemm::VulkanConvolutionIm2ColGemm(VulkanBackend* backend,
                                                         const Convolution2DCommon* common, const float* weightPtr,
                                                         const float* biasPtr, int ci, int co)
    : VulkanConvolutionCommon(common, backend),
      mCi(ci),
      mCo(co),
            mKernelWidth(common->kernelX()) {
    mUseFP16 = backend->useFP16();
        const uint32_t padCi = ALIGN_UP4(static_cast<uint32_t>(ci));
        mPadK = padCi * mKernelWidth * common->kernelY();
    mPadN = ROUND_UP(static_cast<uint32_t>(co), 32u);
    init(backend, common, weightPtr, biasPtr);
}

bool VulkanConvolutionIm2ColGemm::init(VulkanBackend* backend, const Convolution2DCommon* common,
                                       const float* weightPtr, const float* biasPtr) {
    std::vector<VkDescriptorType> packTypes = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    };
    const char* packShader =
        mUseFP16 ? "glsl_convolution_im2col_pack_a_FP16_comp" : "glsl_convolution_im2col_pack_a_comp";
    mPackPipeline = backend->getPipeline(packShader, packTypes);
    if (nullptr == mPackPipeline) {
        return false;
    }
    mPackSet.reset(mPackPipeline->createSet());

    std::vector<VkDescriptorType> gemmTypes = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    };
    uint32_t activation = 0;
    if (common->relu()) {
        activation = 1;
    } else if (common->relu6()) {
        activation = 2;
    }
    const char* gemmShader = mUseFP16 ? "glsl_gemm_m8n4_FP16_comp" : "glsl_gemm_m8n4_comp";
    mGemmPipeline = backend->getPipeline(gemmShader, gemmTypes, {}, {activation});
    if (nullptr == mGemmPipeline) {
        return false;
    }
    mGemmSet.reset(mGemmPipeline->createSet());

    if (nullptr == weightPtr) {
        return true;
    }

    const size_t weightElements = static_cast<size_t>(mPadK) * mPadN;
    std::vector<float> packedWeight(weightElements, 0.0f);
    const int kernelSize = mKernelWidth * common->kernelY();
    const uint32_t padCi = ALIGN_UP4(static_cast<uint32_t>(mCi));
    const uint32_t nBlockCount = mPadN / 32u;
    for (int outputChannel = 0; outputChannel < mCo; ++outputChannel) {
        const uint32_t nBlock = outputChannel / 32;
        const uint32_t nVector = (outputChannel % 32) / 4;
        const uint32_t nComponent = outputChannel % 4;
        for (int inputChannel = 0; inputChannel < mCi; ++inputChannel) {
            for (int kernelIndex = 0; kernelIndex < kernelSize; ++kernelIndex) {
                const uint32_t k = kernelIndex * padCi + inputChannel;
                const uint32_t k4 = k / 4u;
                const uint32_t kComponent = k % 4u;
                const size_t vectorIndex =
                    ((static_cast<size_t>(k4) * nBlockCount + nBlock) * 4u + kComponent) * 8u + nVector;
                const size_t sourceIndex = static_cast<size_t>(outputChannel) * mCi * kernelSize +
                    inputChannel * kernelSize + kernelIndex;
                packedWeight[vectorIndex * 4u + nComponent] = weightPtr[sourceIndex];
            }
        }
    }

    const size_t elementSize = mUseFP16 ? sizeof(int16_t) : sizeof(float);
    mPackedWeight.reset(new VulkanBuffer(backend->getMemoryPool(), false, weightElements * elementSize, nullptr,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         VK_SHARING_MODE_EXCLUSIVE, 0));
    if (mUseFP16) {
        std::vector<int16_t> packedHalf(weightElements);
        FLOAT_TO_HALF(packedWeight.data(), packedHalf.data(), static_cast<int>(weightElements));
        backend->copyToGPUBuffer(packedHalf.data(), mPackedWeight->buffer(), mPackedWeight->size(), 0);
    } else {
        backend->copyToGPUBuffer(packedWeight.data(), mPackedWeight->buffer(), mPackedWeight->size(), 0);
    }

    std::vector<float> bias(mPadN, 0.0f);
    if (nullptr != biasPtr) {
        ::memcpy(bias.data(), biasPtr, static_cast<size_t>(mCo) * sizeof(float));
    }
    mBias.reset(new VulkanBuffer(backend->getMemoryPool(), false, static_cast<size_t>(mPadN) * elementSize, nullptr,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VK_SHARING_MODE_EXCLUSIVE, 0));
    if (mUseFP16) {
        std::vector<int16_t> biasHalf(mPadN);
        FLOAT_TO_HALF(bias.data(), biasHalf.data(), static_cast<int>(mPadN));
        backend->copyToGPUBuffer(biasHalf.data(), mBias->buffer(), mBias->size(), 0);
    } else {
        backend->copyToGPUBuffer(bias.data(), mBias->buffer(), mBias->size(), 0);
    }
    return true;
}

bool VulkanConvolutionIm2ColGemm::valid() const {
    return nullptr != mPackPipeline && nullptr != mGemmPipeline && nullptr != mPackSet && nullptr != mGemmSet &&
        nullptr != mPackedWeight && nullptr != mBias;
}

bool VulkanConvolutionIm2ColGemm::onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto convolution = op->main_as_Convolution2D();
    if (nullptr == convolution || nullptr == convolution->common()) {
        return false;
    }
    auto result = new VulkanConvolutionIm2ColGemm(static_cast<VulkanBackend*>(bn), convolution->common(), nullptr,
                                                  nullptr, mCi, mCo);
    result->mPackedWeight = mPackedWeight;
    result->mBias = mBias;
    if (!result->valid()) {
        delete result;
        return false;
    }
    *dst = result;
    return true;
}

ErrorCode VulkanConvolutionIm2ColGemm::onEncodeConvolution(const Convolution2DCommon* common,
                                                           const std::vector<Tensor*>& inputs,
                                                           const std::vector<Tensor*>& outputs,
                                                           const VulkanCommandPool::Buffer* cmdBuffer,
                                                           const VulkanBuffer*) {
    if (inputs.empty() || outputs.empty() || !valid()) {
        return INVALID_VALUE;
    }
    auto input = inputs[0];
    auto output = outputs[0];
    const uint32_t M = output->batch() * output->height() * output->width();
    if (M == 0u) {
        return NO_ERROR;
    }
    const uint32_t padM = ROUND_UP(M, 64u);
    auto backend = static_cast<VulkanBackend*>(this->backend());
    if (mUseFP16) {
        mPackedInput.reset(Tensor::createDevice<int16_t>({static_cast<int>(padM), static_cast<int>(mPadK)}));
    } else {
        mPackedInput.reset(Tensor::createDevice<float>({static_cast<int>(padM), static_cast<int>(mPadK)}));
    }
    if (!backend->onAcquireBuffer(mPackedInput.get(), Backend::DYNAMIC)) {
        return OUT_OF_MEMORY;
    }

    auto srcBuffer = backend->getTensorBuffer(input);
    auto packedBuffer = backend->getTensorBuffer(mPackedInput.get());
    auto dstBuffer = backend->getTensorBuffer(output);
    const size_t packedSize = backend->getTensorSize(mPackedInput.get());
    auto padding = ConvolutionCommon::convolutionPad(input, output, common);

    struct PackParams {
        uint32_t M;
        uint32_t K;
        uint32_t inputWidth;
        uint32_t inputHeight;
        uint32_t inputChannel4;
        uint32_t outputWidth;
        uint32_t outputHeight;
        uint32_t batch;
        uint32_t kernelWidth;
        uint32_t strideWidth;
        uint32_t strideHeight;
        int32_t padWidth;
        int32_t padHeight;
        uint32_t dilateWidth;
        uint32_t dilateHeight;
    } packParams;
    packParams.M = M;
    packParams.K = mPadK;
    packParams.inputWidth = input->width();
    packParams.inputHeight = input->height();
    packParams.inputChannel4 = UP_DIV(mCi, 4);
    packParams.outputWidth = output->width();
    packParams.outputHeight = output->height();
    packParams.batch = output->batch();
    packParams.kernelWidth = mKernelWidth;
    packParams.strideWidth = common->strideX();
    packParams.strideHeight = common->strideY();
    packParams.padWidth = padding.first;
    packParams.padHeight = padding.second;
    packParams.dilateWidth = common->dilateX();
    packParams.dilateHeight = common->dilateY();

    mPackSet->writeBuffer(srcBuffer.first->buffer(), 0, backend->getTensorSize(input), srcBuffer.second);
    mPackSet->writeBuffer(packedBuffer.first->buffer(), 1, packedSize, packedBuffer.second);
    mPackPipeline->bind(cmdBuffer->get(), mPackSet->get());
    vkCmdPushConstants(cmdBuffer->get(), mPackPipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(packParams),
                       &packParams);
    vkCmdDispatch(cmdBuffer->get(), mPadK / 4u, padM / 64u, 1);
    cmdBuffer->barrierSource(packedBuffer.first->buffer(), packedBuffer.second, packedSize);

    struct GemmParams {
        uint32_t M;
        uint32_t N;
        uint32_t K;
        uint32_t padN;
    } gemmParams;
    gemmParams.M = M;
    gemmParams.N = mCo;
    gemmParams.K = mPadK;
    gemmParams.padN = mPadN;

    mGemmSet->writeBuffer(packedBuffer.first->buffer(), 0, packedSize, packedBuffer.second);
    mGemmSet->writeBuffer(mPackedWeight->buffer(), 1, mPackedWeight->size());
    mGemmSet->writeBuffer(mBias->buffer(), 2, mBias->size());
    mGemmSet->writeBuffer(dstBuffer.first->buffer(), 3, backend->getTensorSize(output), dstBuffer.second);
    mGemmPipeline->bind(cmdBuffer->get(), mGemmSet->get());
    vkCmdPushConstants(cmdBuffer->get(), mGemmPipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gemmParams),
                       &gemmParams);
    vkCmdDispatch(cmdBuffer->get(), mPadN / 32u, padM / 64u, 1);

    backend->onReleaseBuffer(mPackedInput.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

} // namespace MNN
