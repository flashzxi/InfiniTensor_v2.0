#include "core/runtime.h"
#include "operators/LogSoftmax.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"

namespace infini {

// Thread test parameters
template <typename T> struct ThreadTestParams {
    infiniDevice_t device = INFINI_DEVICE_CPU;
    int deviceId = 0;
    Shape shapeInput;
    DataType dataType = DataType(INFINI_DTYPE_F32);
    std::vector<T> inputData;
    std::vector<T> outputData;
    bool completed = false;
    std::string deviceName;
    int dim = 1;  // default dim for log_softmax
};

// Device thread function
template <typename T> void deviceThreadFunc(ThreadTestParams<T> &params) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();

    // Initialize device Context
    runtime->initThreadContext(params.device, params.deviceId);

    // Create Graph
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(params.shapeInput, params.dataType);
    auto op = g->addOp<LogSoftmaxObj>(input, params.dim, nullptr);

    // Set data first (set CPU pointer), then allocate memory (triggers H2D copy)
    input->setData(params.inputData.data());
    runtime->dataMalloc(g);

    // Run computation
    runtime->run(g);

    // Get output and copy to host
    auto output = op->getOutput(0);
    size_t numElements = output->getElement();
    params.outputData.resize(numElements);

    // Check if output data exists
    auto dataBlob = output->getData();
    if (!dataBlob) {
        throw std::runtime_error("Output data blob is null!");
    }
    void *devicePtr = dataBlob->getRawDataPtr();
    if (!devicePtr && !runtime->isCpu()) {
        throw std::runtime_error("Output device pointer is null on GPU device!");
    }

    // Copy result data
    void *hostPtr = runtime->allocHost(output->getTotalBytes());
    runtime->memcpy(hostPtr, devicePtr, output->getTotalBytes(), INFINIRT_MEMCPY_D2H);

    // Use generic function for data copy and conversion
    copyAndConvertData(params.outputData, hostPtr, numElements, params.dataType);

    runtime->deallocHost(hostPtr);
    params.completed = true;
}

// Data generator function type
template <typename T>
using DataGeneratorFunc = std::function<std::vector<T>(size_t, T, T)>;

// Run multi-thread test
template <typename T>
void runMultiThreadTest(
    const Shape &shapeInput, int dim,
    const DataType &dataType,
    DataGeneratorFunc<T> dataGenerator = generateRandomData<T>,
    bool print = false) {

    // Prepare input data
    size_t elementInput = 1;
    for (auto d : shapeInput)
        elementInput *= d;

    // Use the passed data generator function (default uses random data)
    auto inputData = dataGenerator(elementInput, static_cast<T>(-5), static_cast<T>(5));

    // Create thread parameters
    ThreadTestParams<T> cpuParams, gpuParams;

    // CPU thread parameters
    cpuParams.device = INFINI_DEVICE_CPU;
    cpuParams.deviceId = 0;
    cpuParams.shapeInput = shapeInput;
    cpuParams.dataType = dataType;
    cpuParams.inputData = inputData;
    cpuParams.deviceName = "CPU";
    cpuParams.dim = dim;

    // GPU thread parameters
    gpuParams.device = INFINI_DEVICE_NVIDIA;
    gpuParams.deviceId = 0;
    gpuParams.shapeInput = shapeInput;
    gpuParams.dataType = dataType;
    gpuParams.inputData = inputData;
    gpuParams.deviceName = "NVIDIA";
    gpuParams.dim = dim;

    if (print) {
        std::cout << "========================================" << std::endl;
        std::cout << "Running Multi-Thread LogSoftmax Test" << std::endl;
        std::cout << "DataType: " << dataType.toString() << std::endl;
        std::cout << "Shape Input: " << vecToString(shapeInput) << std::endl;
        std::cout << "Dim: " << dim << std::endl;
        std::cout << "Thread 1: CPU (" << dataType.toString() << ")" << std::endl;
        std::cout << "Thread 2: NVIDIA (" << dataType.toString() << ")" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    // Launch two threads for parallel execution
    std::thread cpuThread(deviceThreadFunc<T>, std::ref(cpuParams));
    std::thread gpuThread(deviceThreadFunc<T>, std::ref(gpuParams));

    // Wait for both threads to complete
    cpuThread.join();
    gpuThread.join();

    // Verify results
    ASSERT_TRUE(cpuParams.completed) << "CPU thread failed";
    ASSERT_TRUE(gpuParams.completed) << "NVIDIA thread failed";

    ASSERT_EQ(cpuParams.outputData.size(), gpuParams.outputData.size())
        << "Output size mismatch";

    // Compare results
    size_t numErrors = 0;
    float maxError = 0.0f;
    const float epsilon = 1e-3f;

    for (size_t i = 0; i < cpuParams.outputData.size(); ++i) {
        float cpuVal, gpuVal;

        // Convert to float for comparison
        if constexpr (std::is_same_v<T, float>) {
            cpuVal = cpuParams.outputData[i];
            gpuVal = gpuParams.outputData[i];
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            // FP16 to FP32 comparison
            cpuVal = fp16_to_fp32(cpuParams.outputData[i]);
            gpuVal = fp16_to_fp32(gpuParams.outputData[i]);
        }

        float error = std::abs(cpuVal - gpuVal);
        maxError = std::max(maxError, error);

        if (error > epsilon) {
            numErrors++;
            if (numErrors <= 5) {
                std::cout << "Mismatch at index " << i << ": CPU=" << cpuVal
                          << ", NVIDIA=" << gpuVal << ", error=" << error
                          << std::endl;
            }
        }
    }

    if (print) {
        std::cout << "Result Comparison:" << std::endl;
        std::cout << "  Total elements: " << cpuParams.outputData.size() << std::endl;
        std::cout << "  Errors: " << numErrors << std::endl;
        std::cout << "  Max error: " << maxError << std::endl;

        if (numErrors == 0) {
            std::cout << "  Test PASSED" << std::endl;
        } else {
            std::cout << "  Test FAILED" << std::endl;
        }
        std::cout << "========================================" << std::endl;
    }

    EXPECT_EQ(numErrors, 0) << "Results mismatch between CPU and NVIDIA (max error: "
                              << maxError << ")";
}

// Basic LogSoftmax operation test - F32 with dim=1
TEST(LogSoftmax, LogSoftmax_MultiThread_F32_Dim1) {
    Shape shapeInput = {2, 3, 4};  // 3D input

#ifdef USE_CUDA
    runMultiThreadTest<float>(shapeInput, 1, DataType(INFINI_DTYPE_F32));
#else
    std::cout << "CUDA not enabled, skipping multi-thread test" << std::endl;
#endif
}

// LogSoftmax operation test with dim=-1 (last dim)
TEST(LogSoftmax, LogSoftmax_MultiThread_F32_DimLast) {
    Shape shapeInput = {2, 3, 4, 5};  // 4D input, dim=-1 means dim=3

#ifdef USE_CUDA
    runMultiThreadTest<float>(shapeInput, 3, DataType(INFINI_DTYPE_F32));
#else
    std::cout << "CUDA not enabled, skipping multi-thread test" << std::endl;
#endif
}

// LogSoftmax operation test with dim=0
TEST(LogSoftmax, LogSoftmax_MultiThread_F32_Dim0) {
    Shape shapeInput = {3, 4, 5};  // 3D input

#ifdef USE_CUDA
    runMultiThreadTest<float>(shapeInput, 0, DataType(INFINI_DTYPE_F32));
#else
    std::cout << "CUDA not enabled, skipping multi-thread test" << std::endl;
#endif
}

// Basic LogSoftmax operation test - F16 with dim=1
TEST(LogSoftmax, LogSoftmax_MultiThread_F16_Dim1) {
    Shape shapeInput = {2, 3, 4};  // 3D input

#ifdef USE_CUDA
    runMultiThreadTest<uint16_t>(shapeInput, 1, DataType(INFINI_DTYPE_F16),
                                 generateSequentialData<uint16_t>, true);
#else
    std::cout << "CUDA not enabled, skipping multi-thread test" << std::endl;
#endif
}

// Single device test - CPU
TEST(LogSoftmax, LogSoftmax_SingleDevice_CPU) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_CPU, 0);

    Shape shapeInput = {2, 3, 4};
    int dim = 1;

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<LogSoftmaxObj>(input, dim, nullptr);

    runtime->dataMalloc(g);

    // Set input data
    std::vector<float> inputData(input->getElement());

    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>((i % 10) - 5) * 0.1f; // -0.5 to 0.4
    }

    input->setData(inputData.data());

    // Execute computation
    runtime->run(g);

    // Get output and print
    auto output = op->getOutput(0);
    std::cout << "CPU LogSoftmax Output Data: " << std::endl;
    output->printData(runtime);
}

#ifdef USE_CUDA
// Single device test - NVIDIA F32
TEST(LogSoftmax, LogSoftmax_SingleDevice_NVIDIA_F32) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    Shape shapeInput = {2, 3, 4};
    int dim = 1;

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<LogSoftmaxObj>(input, dim, nullptr);

    // Set input data
    std::vector<float> inputData(input->getElement());

    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>((i % 10) - 5) * 0.1f; // -0.5 to 0.4
    }

    input->setData(inputData.data());
    runtime->dataMalloc(g);

    // Execute computation
    runtime->run(g);

    // Get output and print
    auto output = op->getOutput(0);
    std::cout << "NVIDIA F32 LogSoftmax Output Data: " << std::endl;
    output->printData(runtime);
}

// Single device test - NVIDIA F16
TEST(LogSoftmax, LogSoftmax_SingleDevice_NVIDIA_F16) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    Shape shapeInput = {2, 3, 4};
    int dim = 1;

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F16));
    auto op = g->addOp<LogSoftmaxObj>(input, dim, nullptr);

    // Set input data
    std::vector<uint16_t> inputData(input->getElement());

    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = fp32_to_fp16(static_cast<float>((i % 10) - 5) * 0.1f);
    }

    input->setData(inputData.data());
    runtime->dataMalloc(g);

    // Execute computation
    runtime->run(g);

    // Get output and print
    auto output = op->getOutput(0);
    std::cout << "NVIDIA F16 LogSoftmax Output Data: " << std::endl;
    output->printData(runtime);
}

// Test with different dim values - NVIDIA F32
TEST(LogSoftmax, LogSoftmax_SingleDevice_NVIDIA_F32_Dim2) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    Shape shapeInput = {2, 3, 4, 5};  // 4D input
    int dim = 2;  // Apply log_softmax along dimension 2

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<LogSoftmaxObj>(input, dim, nullptr);

    // Set input data
    std::vector<float> inputData(input->getElement());

    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>((i % 20) - 10) * 0.1f;
    }

    input->setData(inputData.data());
    runtime->dataMalloc(g);

    // Execute computation
    runtime->run(g);

    // Get output and print
    auto output = op->getOutput(0);
    std::cout << "NVIDIA F32 LogSoftmax (dim=2) Output Data: " << std::endl;
    output->printData(runtime);
}
#endif

} // namespace infini
