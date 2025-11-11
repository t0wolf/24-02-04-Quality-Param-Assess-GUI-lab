#include "SpecUnionCls.h"

SpecUnionCls::SpecUnionCls(std::string& sEngineFilePath)
    :m_inputDims({ 1, 3, 512, 512 })
    , m_classes(8)
    , m_outputViewDims({ 1, 8 })
{
    // De-serialize engine from file
    std::ifstream engineFile(sEngineFilePath, std::ios::binary);
    if (engineFile.fail())
    {
        return;
    }

    engineFile.seekg(0, std::ifstream::end);
    auto fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger::gLogger.getTRTLogger());
    m_engine = runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
    assert(m_engine != nullptr);

    m_context = m_engine->createExecutionContext();
    assert(m_context != nullptr);

    m_inputImageIdx = m_engine->getBindingIndex("images");
    m_outputViewIdx = m_engine->getBindingIndex("output_views");

    m_context->setBindingDimensions(m_inputImageIdx, m_inputDims);
    m_context->setBindingDimensions(m_outputViewIdx, m_outputViewDims);
}

SpecUnionCls::~SpecUnionCls()
{
    if (m_engine != nullptr)
    {
        delete m_engine;
        m_engine = nullptr;
    }
    if (m_context != nullptr)
    {
        delete m_context;
        m_context = nullptr;
    }
}

int SpecUnionCls::doInference(cv::Mat& src, std::map<std::string, int>& classResults)
{
    doSingleInfer(src, classResults);
    return 1;
}

int SpecUnionCls::doSingleInfer(cv::Mat& src, std::map<std::string, int>& classResults)
{
    void* inputMem{ nullptr };
    void* outputViewMem{ nullptr };
    size_t inputSize = getMemorySize(m_inputDims, sizeof(float));
    size_t outputViewSize = getMemorySize(m_outputViewDims, sizeof(float));

    if (cudaMalloc(&inputMem, inputSize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: input cuda memory allocation failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }
    if (cudaMalloc(&outputViewMem, outputViewSize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: output cuda memory allocation failed, size = " << outputViewSize << " bytes" << std::endl;
        return 0;
    }

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return 0;
    }

    float* blob = new float[getElementNum(m_inputDims)];
    float* outputView = new float[getElementNum(m_outputViewDims)];

    preprocess(src, blob);

    if (cudaMemcpyAsync(inputMem, blob, inputSize, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of input failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }

    // Run TensorRT inference
    void* bindings[] = { inputMem, outputViewMem };
    bool status = m_context->enqueueV2(bindings, stream, nullptr);
    if (!status)
    {
        logger::gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return 0;
    }

    if (cudaMemcpyAsync(outputView, outputViewMem, outputViewSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << outputViewSize << " bytes" << std::endl;
        return 0;
    }

    postProcess(outputView, classResults);

    delete[] blob;
    delete[] outputView;
    cudaFree(inputMem);
    cudaFree(outputViewMem);
    cudaStreamDestroy(stream);
    return 1;
}

int SpecUnionCls::preprocess(cv::Mat& src, float* blob)
{
    if (src.empty())
        return 0;

    cv::Mat dst;
    src.copyTo(dst);

    cv::Size inputSize(m_inputDims.d[3], m_inputDims.d[2]);  //size(width,height)

    //cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);
    cv::resize(dst, dst, inputSize);
    dst.convertTo(dst, CV_32FC3);
    blobFromImage(dst, blob);
    return 1;
}

int SpecUnionCls::blobFromImage(cv::Mat& src, float* blob)
{
    int channels = src.channels();
    int rows = src.rows;  // cols == width == Point.x;   rows == heigh == Point.y
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

int SpecUnionCls::postProcess(float* outputView, std::map<std::string, int>& classResults)
{
    float maxViewValue = -10000.0f;
    int maxViewIdx = 0;
    for (int i = 0; i < m_outputViewDims.d[1]; ++i)
    {
        float fValue = outputView[i];
        if (fValue > maxViewValue)
        {
            maxViewValue = fValue;
            maxViewIdx = i;
        }
    }
    //auto maxView = std::max_element(outputView, outputView + m_outputViewDims.d[1]);
    //int maxViewIdx = std::distance(outputView, maxView);

    classResults.insert({ "view", maxViewIdx });

    return 1;
}


