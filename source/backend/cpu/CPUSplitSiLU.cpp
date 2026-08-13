//
//  CPUSplitSiLU.cpp
//  MNN
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include <cmath>
#include "CPUBackend.hpp"
#include "MNN_generated.h"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"

namespace MNN {

class CPUSplitSiLU : public Execution {
public:
    CPUSplitSiLU(Backend* backend) : Execution(backend) {
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.size() != 1 || outputs.size() != 1) {
            return NOT_SUPPORT;
        }
        auto input = inputs[0];
        auto output = outputs[0];
        mIsC4 = input->dimensions() == 4 && output->dimensions() == 4 &&
                TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 &&
                TensorUtils::getDescribe(output)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
        if (mIsC4) {
            if (input->length(0) != output->length(0) || input->length(1) != output->length(1) * 2 ||
                input->length(2) != output->length(2) || input->length(3) != output->length(3) ||
                output->length(1) % 4 != 0) {
                return NOT_SUPPORT;
            }
            return NO_ERROR;
        }
        if (input->dimensions() != 3 || output->dimensions() != 3 || input->length(2) != output->length(2) * 2) {
            return NOT_SUPPORT;
        }
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto input = inputs[0];
        auto output = outputs[0];
        const auto inputData = input->host<float>();
        auto outputData = output->host<float>();
        auto cpuBackend = static_cast<CPUBackend*>(backend());
        if (mIsC4) {
            const int batchCount = output->length(0);
            const int channelBlocks = output->length(1) / 4;
            const int spatialSize = output->length(2) * output->length(3);
            const int taskCount = batchCount * channelBlocks * spatialSize;
            const int threadCount = std::min(cpuBackend->threadNumber(), taskCount);
            MNN_CONCURRENCY_BEGIN(threadId, threadCount) {
                const int start = threadId * taskCount / threadCount;
                const int end = (threadId + 1) * taskCount / threadCount;
                for (int index = start; index < end; ++index) {
                    const int spatial = index % spatialSize;
                    const int batch = (index / spatialSize) % batchCount;
                    const int channelBlock = index / (spatialSize * batchCount);
                    const int inputOffset = (channelBlock * batchCount + batch) * spatialSize * 4 + spatial * 4;
                    const int upOffset = inputOffset + channelBlocks * batchCount * spatialSize * 4;
                    for (int lane = 0; lane < 4; ++lane) {
                        const float gate = inputData[inputOffset + lane];
                        const float up = inputData[upOffset + lane];
                        outputData[inputOffset + lane] = up * gate / (1.0f + std::exp(-gate));
                    }
                }
            }
            MNN_CONCURRENCY_END();
            return NO_ERROR;
        }
        const int hiddenSize = output->length(2);
        const int rowCount = output->elementSize() / hiddenSize;
        const int threadCount = std::min(cpuBackend->threadNumber(), rowCount);

        MNN_CONCURRENCY_BEGIN(threadId, threadCount) {
            const int start = threadId * rowCount / threadCount;
            const int end = (threadId + 1) * rowCount / threadCount;
            for (int row = start; row < end; ++row) {
                const auto inputOffset = row * hiddenSize * 2;
                const auto outputOffset = row * hiddenSize;
                for (int channel = 0; channel < hiddenSize; ++channel) {
                    const float gate = inputData[inputOffset + channel];
                    const float up = inputData[inputOffset + hiddenSize + channel];
                    outputData[outputOffset + channel] = up * gate / (1.0f + std::exp(-gate));
                }
            }
        }
        MNN_CONCURRENCY_END();
        return NO_ERROR;
    }

private:
    bool mIsC4 = false;
};

class CPUSplitSiLUCreator : public CPUBackend::Creator {
public:
    Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                        Backend* backend) const override {
        auto cpuBackend = static_cast<CPUBackend*>(backend);
        if (cpuBackend->functions()->bytes != sizeof(float)) {
            return nullptr;
        }
        for (auto input : inputs) {
            TensorUtils::setTensorSupportPack(input, false);
        }
        for (auto output : outputs) {
            TensorUtils::setTensorSupportPack(output, false);
        }
        return new CPUSplitSiLU(backend);
    }
};

REGISTER_CPU_OP_CREATOR_TRANSFORMER(CPUSplitSiLUCreator, OpType_SplitSiLU);

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE