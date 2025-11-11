#include "view_cls_inferer.h"

ViewClsInferer::ViewClsInferer(std::string backbonePath, std::string swinHeadPath)
{
    std::ifstream backboneEngineFile, swinHeadEngineFile;
    std::vector<char> backboneEngineData, swinHeadEngineData;
    auto backboneFsize = parseEngineFile(backbonePath, backboneEngineFile, backboneEngineData);
    auto swinHeadFsize = parseEngineFile(swinHeadPath, swinHeadEngineFile, swinHeadEngineData);
    // backboneEngineFile.read(backboneEngineData.data(), backboneEngineFile.tellg());
    // swinHeadEngineFile.read(swinHeadEngineData.data(), swinHeadEngineFile.tellg());

    // Backbone Engine building
    auto& inferLogger = logger::gLogger.getTRTLogger();
    //inferLogger.setReportableSeverity();
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger::gLogger.getTRTLogger());
    //nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger::gLogger.getTRTLogger());
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(backboneEngineData.data(), backboneFsize, nullptr);
    assert(engine != nullptr);

    m_backboneContext = engine->createExecutionContext();
    assert(m_backboneContext != nullptr);

    m_backboneInputIdx = engine->getBindingIndex("input.1");
    //m_backboneOutputIdx = engine->getBindingIndex("338");
    m_backboneOutputIdx = engine->getBindingIndex("490");

    m_backboneContext->setBindingDimensions(m_backboneInputIdx, m_backboneInputDims);

    // Swin Head Engine building
    runtime = nvinfer1::createInferRuntime(logger::gLogger.getTRTLogger());
    engine = runtime->deserializeCudaEngine(swinHeadEngineData.data(), swinHeadFsize, nullptr);
    assert(engine != nullptr);

    m_swinHeadContext = engine->createExecutionContext();
    assert(m_swinHeadContext != nullptr);

    m_swinHeadInputIdx = engine->getBindingIndex("onnx::Unsqueeze_0");
    //m_swinHeadOutputIdx = engine->getBindingIndex("838");
    m_swinHeadOutputIdx = engine->getBindingIndex("958");

    m_swinHeadContext->setBindingDimensions(m_swinHeadInputIdx, m_swinHeadInputDims);

    m_clipSize = m_swinHeadInputDims.d[0];
}

std::streampos ViewClsInferer::parseEngineFile(std::string& enginePath, std::ifstream& engineFile, std::vector<char>& engineData)
{
    engineFile.open(enginePath, std::ios::binary);
    // if (engineFile.fail())
    // {
    //     return ;
    // }

    engineFile.seekg(0, std::ifstream::end);
    auto fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    engineData = std::vector<char>(fsize);
    engineFile.read(engineData.data(), fsize);
    return fsize;
}

int ViewClsInferer::doInference(cv::Mat& src)
{
    cv::Mat inferFrame = src.clone();
    float* feat = new float[getElementNum(m_backboneInputDims)];
    backboneSingleInfer(inferFrame, feat);
    m_vFeats.push_back(feat);

    if (m_vFeats.size() == m_clipSize)
    {
        float* probs = new float[m_classes];
        swinHeadSingleInfer(probs);
        int viewIdx = postProcess(probs);
        clearAndReleaseFeats();

        delete[] probs;
        probs = nullptr;

        return viewIdx;
    }
    else
    {
        return -1;
    }
}

int ViewClsInferer::doInference(cv::Mat& src, float& fQualityScore)
{
    cv::Mat inferFrame = src.clone();
    float* feat = new float[getElementNum(m_backboneInputDims)];
    backboneSingleInfer(inferFrame, feat);
    m_vFeats.push_back(feat);

    if (m_vFeats.size() == m_clipSize)
    {
        float* probs = new float[m_classes];
        float* pQualityScore = new float[1];
        swinHeadSingleInfer(probs, pQualityScore);

        fQualityScore = *pQualityScore;

        int viewIdx = postProcess(probs);
        clearAndReleaseFeats();

        delete[] probs;
        delete[] pQualityScore;

        return viewIdx;
    }
    else
    {
        return -1;
    }
}

int ViewClsInferer::doInference(cv::Mat& src, float& fQualityScore, bool& bIsSwitch)
{
    cv::Mat inferFrame = src.clone();
    float* feat = new float[getElementNum(m_backboneInputDims)];
    backboneSingleInfer(inferFrame, feat);
    m_vFeats.push_back(feat);

    if (m_vFeats.size() == m_clipSize)
    {
        float* probs = new float[m_classes];
        float* pQualityScore = new float[1];
        float* pIsSwitch = new float[2];
        swinHeadSingleInfer(probs, pQualityScore, pIsSwitch);

        fQualityScore = *pQualityScore;

        int viewIdx = postProcess(probs);

        int switchIdx = postProcess(pIsSwitch, 2);
        bIsSwitch = switchIdx == 1 ? true : false;
        clearAndReleaseFeats();

        delete[] probs;
        delete[] pQualityScore;
        delete[] pIsSwitch;

        return viewIdx;
    }
    else
    {
        return -1;
    }
}

int ViewClsInferer::clearFeatMemory()
{
    clearAndReleaseFeats();
    return 1;
}

int ViewClsInferer::getClipSize()
{
    return m_clipSize;
}

int ViewClsInferer::postProcess(float* probs)
{
    auto resultPair = argmax(probs, m_classes);

    return resultPair.second;
}

int ViewClsInferer::postProcess(float* probs, int length)
{
    auto resultPair = argmax(probs, length);

    return resultPair.second;
}

int ViewClsInferer::blobFromImage(cv::Mat& src, float* blob)
{
    int channels = src.channels();
    int rows = src.rows;
    int cols = src.cols;

    for (int c = 0; c < channels; c++)
    {
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                blob[c * rows * cols + row * cols + col] = (src.at<cv::Vec3f>(row, col)[c] / 255.0f - m_means[c]) / m_stds[c];
            }
        }
    }
    return 1;
}

int ViewClsInferer::blobFromFeatures(std::vector<float*> vBlobs, float* blob)
{
    int counter = 0;
    for (auto& blob_ : vBlobs)
    {
        for (int i = 0; i < m_swinInputSingleSize; i++)
        {
            blob[counter * m_swinInputSingleSize + i] = blob_[i];
            // std::cout << blob_[i] << " ";
        }

        counter++;
    }
    return 1;
}

int ViewClsInferer::preprocess(cv::Mat& src, float* blob)
{
    cv::Mat dst;
    src.copyTo(dst);

    cv::Size inputSize(m_backboneInputDims.d[2], m_backboneInputDims.d[3]);

    //cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);
    cv::resize(dst, dst, inputSize);
    dst.convertTo(dst, CV_32FC3);
    blobFromImage(dst, blob);
    return 1;
}

int ViewClsInferer::backboneSingleInfer(cv::Mat& src, float* feat)
{
    auto start = std::chrono::system_clock::now();
    void* inputMem{ nullptr };
    void* outputMem{ nullptr };
    size_t inputSize = getMemorySize(m_backboneInputDims, sizeof(float));
    size_t outputSize = getMemorySize(m_backboneOutputDims, sizeof(float));
    // int outputSize = 51 * sizeof(float);

    if (cudaMalloc(&inputMem, inputSize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: input cuda memory allocation failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }
    if (cudaMalloc(&outputMem, outputSize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: output cuda memory allocation failed, size = " << outputSize << " bytes" << std::endl;
        return 0;
    }

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return 0;
    }

    float* blob = new float[getElementNum(m_backboneInputDims)];
    preprocess(src, blob);

    if (cudaMemcpyAsync(inputMem, blob, inputSize, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of input failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }

    // // Run TensorRT inference
    void* bindings[] = { inputMem, outputMem };
    bool status = m_backboneContext->enqueueV2(bindings, stream, nullptr);
    if (!status)
    {
        logger::gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return 0;
    }

    if (cudaMemcpyAsync(feat, outputMem, outputSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << outputSize << " bytes" << std::endl;
        return 0;
    }

    delete[] blob;
    cudaFree(inputMem);
    cudaFree(outputMem);
    cudaStreamDestroy(stream);
    auto end = std::chrono::system_clock::now();
    // std::cout << "[I] Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    // std::cout << "backboneSingleInfer" <<std::endl;

    return 1;
}

int ViewClsInferer::swinHeadSingleInfer(float* probs)
{
    auto start = std::chrono::system_clock::now();
    void* inputMem{ nullptr };
    void* outputClsMem{ nullptr };
    void* outputQualityMem{ nullptr };

    size_t inputSize = getMemorySize(m_swinHeadInputDims, sizeof(float));
    size_t outputSize = m_classes * sizeof(float);

    if (cudaMalloc(&inputMem, inputSize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: input cuda memory allocation failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }
    if (cudaMalloc(&outputClsMem, outputSize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: output cuda memory allocation failed, size = " << outputSize << " bytes" << std::endl;
        return 0;
    }

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return 0;
    }

    float* blob = new float[getElementNum(m_swinHeadInputDims)];
    blobFromFeatures(m_vFeats, blob);

    if (cudaMemcpyAsync(inputMem, blob, inputSize, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of input failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }

    // Run TensorRT inference
    void* bindings[] = { inputMem, outputClsMem };
    bool status = m_swinHeadContext->enqueueV2(bindings, stream, nullptr);
    if (!status)
    {
        logger::gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return 0;
    }

    if (cudaMemcpyAsync(probs, outputClsMem, outputSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << outputSize << " bytes" << std::endl;
        return 0;
    }

    std::vector<float> vecProbs;
    for (int i = 0; i < 10; ++i)
        vecProbs.push_back(probs[i]);

    delete[] blob;
    cudaFree(inputMem);
    cudaFree(outputClsMem);
    cudaStreamDestroy(stream);
    auto end = std::chrono::system_clock::now();
    // std::cout << "[I] Swin and Head Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    return 1;
}

int ViewClsInferer::swinHeadSingleInfer(float* probs, float* qualityScore)
{
    void* inputMem{ nullptr };
    void* outputClsMem{ nullptr };
    void* outputQualityMem{ nullptr };

    size_t inputSize = getMemorySize(m_swinHeadInputDims, sizeof(float));
    size_t outputSize = m_classes * sizeof(float);
    size_t qualitySize = 1 * sizeof(float);

    if (cudaMalloc(&inputMem, inputSize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: input cuda memory allocation failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }
    if (cudaMalloc(&outputClsMem, outputSize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: output cuda memory allocation failed, size = " << outputSize << " bytes" << std::endl;
        return 0;
    }
    if (cudaMalloc(&outputQualityMem, qualitySize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: output cuda memory allocation failed, size = " << qualitySize << " bytes" << std::endl;
        return 0;
    }

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return 0;
    }

    float* blob = new float[getElementNum(m_swinHeadInputDims)];
    blobFromFeatures(m_vFeats, blob);

    if (cudaMemcpyAsync(inputMem, blob, inputSize, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of input failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }

    // Run TensorRT inference
    void* bindings[] = { inputMem, outputClsMem, outputQualityMem };
    bool status = m_swinHeadContext->enqueueV2(bindings, stream, nullptr);
    if (!status)
    {
        logger::gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return 0;
    }

    if (cudaMemcpyAsync(probs, outputClsMem, outputSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << outputSize << " bytes" << std::endl;
        return 0;
    }

    if (cudaMemcpyAsync(qualityScore, outputQualityMem, qualitySize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << qualitySize << " bytes" << std::endl;
        return 0;
    }

    std::vector<float> vecProbs;
    for (int i = 0; i < m_classes; ++i)
        vecProbs.push_back(probs[i]);

    delete[] blob;
    cudaFree(inputMem);
    cudaFree(outputClsMem);
    cudaFree(outputQualityMem);
    cudaStreamDestroy(stream);
    return 1;
}

int ViewClsInferer::swinHeadSingleInfer(float* probs, float* qualityScore, float* isSwitch)
{
    void* inputMem{ nullptr };
    void* outputClsMem{ nullptr };
    void* outputQualityMem{ nullptr };
    void* outputSwitchMem{ nullptr };

    size_t inputSize = getMemorySize(m_swinHeadInputDims, sizeof(float));
    size_t outputSize = m_classes * sizeof(float);
    size_t qualitySize = 1 * sizeof(float);
    size_t switchSize = 2 * sizeof(float);

    if (cudaMalloc(&inputMem, inputSize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: input cuda memory allocation failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }
    if (cudaMalloc(&outputClsMem, outputSize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: output cuda memory allocation failed, size = " << outputSize << " bytes" << std::endl;
        return 0;
    }
    if (cudaMalloc(&outputQualityMem, qualitySize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: output cuda memory allocation failed, size = " << qualitySize << " bytes" << std::endl;
        return 0;
    }
    if (cudaMalloc(&outputSwitchMem, switchSize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: output cuda memory allocation failed, size = " << switchSize << " bytes" << std::endl;
        return 0;
    }

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return 0;
    }

    float* blob = new float[getElementNum(m_swinHeadInputDims)];
    blobFromFeatures(m_vFeats, blob);

    if (cudaMemcpyAsync(inputMem, blob, inputSize, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of input failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }

    // Run TensorRT inference
    void* bindings[] = { inputMem, outputClsMem, outputQualityMem, outputSwitchMem };
    bool status = m_swinHeadContext->enqueueV2(bindings, stream, nullptr);
    if (!status)
    {
        logger::gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return 0;
    }

    if (cudaMemcpyAsync(probs, outputClsMem, outputSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << outputSize << " bytes" << std::endl;
        return 0;
    }

    if (cudaMemcpyAsync(qualityScore, outputQualityMem, qualitySize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << qualitySize << " bytes" << std::endl;
        return 0;
    }

    if (cudaMemcpyAsync(isSwitch, outputSwitchMem, switchSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << switchSize << " bytes" << std::endl;
        return 0;
    }

    std::vector<float> vecProbs;
    for (int i = 0; i < m_classes; ++i)
        vecProbs.push_back(probs[i]);

    delete[] blob;
    cudaFree(inputMem);
    cudaFree(outputClsMem);
    cudaFree(outputQualityMem);
    cudaFree(outputSwitchMem);
    cudaStreamDestroy(stream);
    return 1;
}
