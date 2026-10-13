#include <cstdlib>

// API 可见性
#define API extern "C" __attribute__((visibility("default")))

size_t get_dtype_size(ncclDataType_t datatype) {
    switch (datatype) {
        // 1 Byte Types
        case ncclInt8:       // 0
        case ncclUint8:      // 1
        case ncclFloat8e4m3: // 10 (FP8)
        case ncclFloat8e5m2: // 11 (FP8)
            return 1;
        // 2 Byte Types
        case ncclFloat16:    // 6 (Half)
        case ncclBfloat16:   // 9 (BF16)
            return 2;
        // 4 Byte Types
        case ncclInt32:      // 2
        case ncclUint32:     // 3
        case ncclFloat32:    // 7 (Float)
            return 4;
        // 8 Byte Types
        case ncclInt64:      // 4
        case ncclUint64:     // 5
        case ncclFloat64:    // 8 (Double)
            return 8;
        default:
            // 遇到未知类型，通常返回 0 或打印错误
            // 为了防止除以0或死循环，返回 1 是个安全的兜底策略，或者直接报错
            return 4;
    }
}
