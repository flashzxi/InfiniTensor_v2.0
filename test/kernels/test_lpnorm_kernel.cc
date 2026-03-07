#include "core/runtime.h"
#include "operators/LpNorm.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"

namespace infini {

// Single device test parameters
template <typename T> struct TestParams {
    infiniDevice_t device = INFINI_DEVICE_NVIDIA;
    int deviceId = 0;
    Shape shapeInput;
    DataType dataType = DataType(INFINI_DTYPE_F32);
    std::vector<T> inputData;
    std::vector<T> outputData;
    bool completed = false;
    std::string deviceName;
    int axis = 1;       // default axis for lp_norm
    int p = 2;          // default p for L2 norm
    float eps = 1e-12f; // default epsilon
};

// Device thread function for single GPU test
template <typename T> void deviceTestFunc(TestParams<T> &params) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();

    // Initialize device Context
    runtime->initThreadContext(params.device, params.deviceId);

    // Create Graph
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(params.shapeInput, params.dataType);
    auto op = g->addOp<LpNormObj>(input, params.axis, params.p, params.eps, nullptr);

    // Set input data (CPU pointer) BEFORE dataMalloc to skip GPU allocation
    input->setData(params.inputData.data());
    // Allocate memory (output only, input is externally managed)
    runtime->dataMalloc(g);

    // Manually copy input data from CPU to GPU device memory
    void *deviceInput = runtime->allocDevice(input->getTotalBytes());
    runtime->memcpy(deviceInput, params.inputData.data(), input->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    input->setData(deviceInput);

    // Run computation
    runtime->run(g);

    // Synchronize to ensure computation is complete before copying data
    runtime->synchronize();

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
    if (!devicePtr) {
        throw std::runtime_error("Output device pointer is null on GPU device!");
    }

    // Copy result data from device to host
    void *hostPtr = runtime->allocHost(output->getTotalBytes());
    runtime->memcpy(hostPtr, devicePtr, output->getTotalBytes(), INFINIRT_MEMCPY_D2H);
    copyAndConvertData(params.outputData, hostPtr, numElements, params.dataType);
    runtime->deallocHost(hostPtr);

    // Clean up device memory
    runtime->deallocDevice(deviceInput);
    runtime->deallocDevice(devicePtr);
    // Note: Do NOT clean up workspace here, as it will break subsequent tests
    // The workspace is initialized once per thread context and reused across tests

    params.completed = true;
}

// Data generator function type
template <typename T>
using DataGeneratorFunc = std::function<std::vector<T>(size_t, T, T)>;

// Run single GPU test
template <typename T>
void runGpuTest(
    const Shape &shapeInput, int axis, int p, float eps,
    const DataType &dataType, int deviceId = 0,
    DataGeneratorFunc<T> dataGenerator = generateRandomData<T>,
    bool print = false) {

    // Prepare input data
    size_t elementInput = 1;
    for (auto d : shapeInput)
        elementInput *= d;

    // Use the passed data generator function (default uses random data)
    auto inputData = dataGenerator(elementInput, static_cast<T>(-5), static_cast<T>(5));

    // Create test parameters
    TestParams<T> gpuParams;

    // GPU parameters
    gpuParams.device = INFINI_DEVICE_NVIDIA;
    gpuParams.deviceId = deviceId;
    gpuParams.shapeInput = shapeInput;
    gpuParams.dataType = dataType;
    gpuParams.inputData = inputData;
    gpuParams.deviceName = "NVIDIA";
    gpuParams.axis = axis;
    gpuParams.p = p;
    gpuParams.eps = eps;

    if (print) {
        std::cout << "========================================" << std::endl;
        std::cout << "Running LpNorm Test on NVIDIA GPU" << std::endl;
        std::cout << "DataType: " << dataType.toString() << std::endl;
        std::cout << "Shape Input: " << vecToString(shapeInput) << std::endl;
        std::cout << "Axis: " << axis << ", p: " << p << ", eps: " << eps << std::endl;
        std::cout << "========================================" << std::endl;
    }

    // Run test
    deviceTestFunc<T>(gpuParams);

    // Verify results
    ASSERT_TRUE(gpuParams.completed) << "GPU test failed";
    ASSERT_EQ(gpuParams.outputData.size(), elementInput) << "Output size mismatch";

    if (print) {
        std::cout << "Result:" << std::endl;
        std::cout << "  Total elements: " << gpuParams.outputData.size() << std::endl;
        std::cout << "  Test PASSED" << std::endl;
        std::cout << "========================================" << std::endl;
    }
}

// Basic LpNorm operation test - F32 with L2 norm
TEST(LpNorm, LpNorm_GPU_F32_L2) {
#ifdef USE_CUDA
    Shape shapeInput = {2, 3, 4};  // 3D input
    runGpuTest<float>(shapeInput, 1, 2, 1e-12f, DataType(INFINI_DTYPE_F32));
#else
    std::cout << "CUDA not enabled, skipping test" << std::endl;
#endif
}

// LpNorm operation test with L1 norm
TEST(LpNorm, LpNorm_GPU_F32_L1) {
#ifdef USE_CUDA
    Shape shapeInput = {3, 4, 5};  // 3D input
    runGpuTest<float>(shapeInput, 1, 1, 1e-12f, DataType(INFINI_DTYPE_F32));
#else
    std::cout << "CUDA not enabled, skipping test" << std::endl;
#endif
}

// Basic LpNorm operation test - F16 with L2 norm
TEST(LpNorm, LpNorm_GPU_F16_L2) {
#ifdef USE_CUDA
    Shape shapeInput = {2, 3, 4};  // 3D input
    runGpuTest<uint16_t>(shapeInput, 1, 2, 1e-12f, DataType(INFINI_DTYPE_F16),
                         0, generateSequentialData<uint16_t>, true);
#else
    std::cout << "CUDA not enabled, skipping test" << std::endl;
#endif
}

#ifdef USE_CUDA
// Single device test - NVIDIA F32
TEST(LpNorm, LpNorm_SingleDevice_NVIDIA_F32) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    Shape shapeInput = {2, 3, 4};
    int axis = 1;
    int p = 2;
    float eps = 1e-12f;

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<LpNormObj>(input, axis, p, eps, nullptr);

    // Set input data
    std::vector<float> inputData(input->getElement());

    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>((i % 10) - 5) * 0.1f; // -0.5 to 0.4
    }

    // Set input data (CPU pointers) BEFORE dataMalloc to skip GPU allocation
    input->setData(inputData.data());
    // Allocate memory (output only, input is externally managed)
    runtime->dataMalloc(g);

    // Manually copy input data from CPU to GPU device memory
    void *deviceInput = runtime->allocDevice(input->getTotalBytes());
    runtime->memcpy(deviceInput, inputData.data(), input->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    input->setData(deviceInput);

    // Execute computation
    runtime->run(g);

    // Synchronize to ensure computation is complete before printing
    runtime->synchronize();

    // Get output and print
    auto output = op->getOutput(0);
    std::cout << "NVIDIA F32 LpNorm Output Data: " << std::endl;
    output->printData(runtime);

    // Clean up device memory
    runtime->deallocDevice(deviceInput);
    // Clean up output memory allocated by dataMalloc
    auto outputData = output->getData();
    if (outputData) {
        runtime->deallocDevice(outputData->getRawDataPtr());
    }
}

// Single device test - NVIDIA F16
TEST(LpNorm, LpNorm_SingleDevice_NVIDIA_F16) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    Shape shapeInput = {2, 3, 4};
    int axis = 1;
    int p = 2;
    float eps = 1e-12f;

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F16));
    auto op = g->addOp<LpNormObj>(input, axis, p, eps, nullptr);

    // Set input data
    std::vector<uint16_t> inputData(input->getElement());

    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = fp32_to_fp16(static_cast<float>((i % 10) - 5) * 0.1f);
    }

    // Set input data (CPU pointers) BEFORE dataMalloc to skip GPU allocation
    input->setData(inputData.data());
    // Allocate memory (output only, input is externally managed)
    runtime->dataMalloc(g);

    // Manually copy input data from CPU to GPU device memory
    void *deviceInput = runtime->allocDevice(input->getTotalBytes());
    runtime->memcpy(deviceInput, inputData.data(), input->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    input->setData(deviceInput);

    // Execute computation
    runtime->run(g);

    // Synchronize to ensure computation is complete before printing
    runtime->synchronize();

    // Get output and print
    auto output = op->getOutput(0);
    std::cout << "NVIDIA F16 LpNorm Output Data: " << std::endl;
    output->printData(runtime);

    // Clean up device memory
    runtime->deallocDevice(deviceInput);
    // Clean up output memory allocated by dataMalloc
    auto outputData = output->getData();
    if (outputData) {
        runtime->deallocDevice(outputData->getRawDataPtr());
    }
}

// Test with L1 norm - NVIDIA F32
TEST(LpNorm, LpNorm_SingleDevice_NVIDIA_F32_L1) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    Shape shapeInput = {4, 5};  // 2D input
    int axis = 1;  // Apply lp_norm along dimension 1
    int p = 1;     // L1 norm
    float eps = 1e-12f;

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<LpNormObj>(input, axis, p, eps, nullptr);

    // Set input data
    std::vector<float> inputData(input->getElement());

    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>(i + 1) * 0.1f;
    }

    // Set input data (CPU pointers) BEFORE dataMalloc to skip GPU allocation
    input->setData(inputData.data());
    // Allocate memory (output only, input is externally managed)
    runtime->dataMalloc(g);

    // Manually copy input data from CPU to GPU device memory
    void *deviceInput = runtime->allocDevice(input->getTotalBytes());
    runtime->memcpy(deviceInput, inputData.data(), input->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    input->setData(deviceInput);

    // Execute computation
    runtime->run(g);

    // Synchronize to ensure computation is complete before printing
    runtime->synchronize();

    // Get output and print
    auto output = op->getOutput(0);
    std::cout << "NVIDIA F32 LpNorm (L1, axis=1) Output Data: " << std::endl;
    output->printData(runtime);

    // Clean up device memory
    runtime->deallocDevice(deviceInput);
    // Clean up output memory allocated by dataMalloc
    auto outputData = output->getData();
    if (outputData) {
        runtime->deallocDevice(outputData->getRawDataPtr());
    }
}
#endif

} // namespace infini
