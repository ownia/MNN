//
//  SplitSiLUTest.cpp
//  MNNTests
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <cmath>
#include <cstring>
#include <vector>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include "TestUtils.h"

using namespace MNN;
using namespace MNN::Express;

static VARP _SplitSiLU(VARP input) {
    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_SplitSiLU;
    return Variable::create(Expr::create(std::move(op), {input}));
}

class SplitSiLUTest : public MNNTestCase {
public:
    virtual ~SplitSiLUTest() = default;

    bool runCase(int batch, int sequence, int hiddenSize) {
        std::vector<float> inputData(batch * sequence * hiddenSize * 2);
        std::vector<float> expected(batch * sequence * hiddenSize);
        for (int row = 0; row < batch * sequence; ++row) {
            const int inputOffset = row * hiddenSize * 2;
            const int outputOffset = row * hiddenSize;
            for (int channel = 0; channel < hiddenSize; ++channel) {
                const float gate = (float)((row * hiddenSize + channel) % 11 - 5) * 0.35f;
                const float up = (float)((row * hiddenSize + channel * 3) % 13 - 6) * 0.2f;
                inputData[inputOffset + channel] = gate;
                inputData[inputOffset + hiddenSize + channel] = up;
                expected[outputOffset + channel] = up * gate / (1.0f + std::exp(-gate));
            }
        }

        auto input = _Input({batch, sequence, hiddenSize * 2}, NCHW, halide_type_of<float>());
        ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
        input->unMap();
        auto output = _SplitSiLU(input);
        auto info = output->getInfo();
        if (info == nullptr || info->dim != std::vector<int>({batch, sequence, hiddenSize})) {
            MNN_ERROR("SplitSiLU shape test failed.\n");
            return false;
        }
        if (!checkVector<float>(output->readMap<float>(), expected.data(), expected.size(), 1e-6f)) {
            MNN_ERROR("SplitSiLU value test failed.\n");
            return false;
        }
        return true;
    }

    bool runC4Case() {
        const int batch = 2;
        const int hiddenSize = 8;
        const int height = 2;
        const int width = 3;
        std::vector<float> inputData(batch * hiddenSize * 2 * height * width);
        std::vector<float> expected(batch * hiddenSize * height * width);
        for (int n = 0; n < batch; ++n) {
            for (int channel = 0; channel < hiddenSize; ++channel) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        const int spatial = h * width + w;
                        const int gateIndex = ((n * hiddenSize * 2 + channel) * height + h) * width + w;
                        const int upIndex = ((n * hiddenSize * 2 + hiddenSize + channel) * height + h) * width + w;
                        const int outputIndex = ((n * hiddenSize + channel) * height + h) * width + w;
                        const float gate = (float)((n * hiddenSize + channel + spatial) % 11 - 5) * 0.35f;
                        const float up = (float)((n * hiddenSize + channel * 3 + spatial) % 13 - 6) * 0.2f;
                        inputData[gateIndex] = gate;
                        inputData[upIndex] = up;
                        expected[outputIndex] = up * gate / (1.0f + std::exp(-gate));
                    }
                }
            }
        }

        auto input = _Input({batch, hiddenSize * 2, height, width}, NCHW, halide_type_of<float>());
        ::memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
        input->unMap();
        auto output = _SplitSiLU(_Convert(input, NC4HW4));
        auto info = output->getInfo();
        if (info == nullptr || info->dim != std::vector<int>({batch, hiddenSize, height, width}) || info->order != NC4HW4) {
            MNN_ERROR("SplitSiLU C4 shape test failed.\n");
            return false;
        }
        auto logicalOutput = _Convert(output, NCHW);
        if (!checkVector<float>(logicalOutput->readMap<float>(), expected.data(), expected.size(), 1e-5f)) {
            MNN_ERROR("SplitSiLU C4 value test failed.\n");
            return false;
        }
        return true;
    }

    bool run(int precision) override {
        return runCase(1, 1, 1) && runCase(1, 3, 4) && runCase(2, 2, 8) && runC4Case();
    }
};

MNNTestSuiteRegister(SplitSiLUTest, "op/split_silu");

#endif // MNN_SUPPORT_TRANSFORMER_FUSE