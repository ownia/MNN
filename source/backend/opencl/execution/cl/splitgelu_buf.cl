#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__kernel void splitgelu_buf(__private int global_dim0, __private int global_dim1,
                        __global const FLOAT * input,
                        #ifdef DOUBLE_INPUTS
                        __global const FLOAT * input1,
                        #endif
                        __global FLOAT * output,
                        __private const int4 shape
){
#ifdef SPLIT_SILU_C4
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        const int spatialSize = shape.y * shape.z;
        const int batch = pos.y / spatialSize;
        const int spatial = pos.y % spatialSize;
        const int channelBlocks = shape.w / 4;
        const int inputOffset = (pos.x * shape.x + batch) * spatialSize * 4 + spatial * 4;
        const int outputOffset = (pos.x * shape.x + batch) * spatialSize * 4 + spatial * 4;
        float4 gate = convert_float4(vload4(0, input + inputOffset));
        float4 up = convert_float4(vload4(0, input + inputOffset + channelBlocks * shape.x * spatialSize * 4));
        float4 out = gate * native_recip((float4)1.0 + native_exp(-gate));
        vstore4(CONVERT_FLOAT4(out * up), 0, output + outputOffset);
    }
#else
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        const int h   = pos.x;
        const int bc  = pos.y;

// The product of W and H is a multiple of 16
#ifdef WH_16
    const int in_offset = bc * shape.z * 2 + h * 16;
    const int out_offset = bc * shape.z + h * 16;

    float16 valueL = convert_float16(vload16(0, input + in_offset));
    float16 valueR = convert_float16(vload16(0, input + in_offset + shape.z));

    #ifdef DOUBLE_INPUTS
    float16 valueConstL = convert_float16(vload16(h, input1));
    float16 valueConstR = convert_float16(vload16(h, input1 + shape.z));
    valueL += valueConstL;
    valueR += valueConstR;
    #endif
    #ifdef SPLIT_SILU
    float16 out = valueL * native_recip((float16)1.0 + native_exp(-valueL));
    out *= valueR;
    #else
    float16 out = (erf(valueR * (float16)0.7071067932881648) + (float16)1.0) * valueR * (float16)0.5;
    out *= valueL;
    #endif
    vstore16(CONVERT_FLOAT16(out), 0, output + out_offset);

// The product of W and H is a multiple of 4
#elif defined (WH_4)

    const int in_offset = bc * shape.z * 2 + h * 4;
    const int out_offset = bc * shape.z + h * 4;

    float4 valueL = convert_float4(vload4(0, input + in_offset));
    float4 valueR = convert_float4(vload4(0, input + in_offset + shape.z));

    #ifdef DOUBLE_INPUTS
    float4 valueConstL = convert_float4(vload4(h, input1));
    float4 valueConstR = convert_float4(vload4(h, input1 + shape.z));
    valueL += valueConstL;
    valueR += valueConstR;
    #endif
    #ifdef SPLIT_SILU
    float4 out = valueL * native_recip((float4)1.0 + native_exp(-valueL));
    out *= valueR;
    #else
    float4 out = (erf(valueR * (float4)0.7071067932881648) + (float4)1.0) * valueR * (float4)0.5;
    out *= valueL;
    #endif
    vstore4(CONVERT_FLOAT4(out), 0, output + out_offset);
#else
    const int in_offset = bc * shape.z * 2 + h;
    const int out_offset = bc * shape.z + h;
    
    float valueL = (float)input[in_offset];
    float valueR = (float)input[in_offset + shape.z];

    #ifdef DOUBLE_INPUTS
    float valueConstL = input1[h];
    float valueConstR = input1[shape.z+h];
    valueL += valueConstL;
    valueR += valueConstR;
    #endif
    #ifdef SPLIT_SILU
    float out = valueL * native_recip(1.0f + native_exp(-valueL));
    out *= valueR;
    #else
    float out = (erf(valueR * 0.7071067932881648) + 1.0) * valueR * 0.5;
    out *= valueL;
    #endif
    output[out_offset] = out;
#endif
    }
#endif
}
