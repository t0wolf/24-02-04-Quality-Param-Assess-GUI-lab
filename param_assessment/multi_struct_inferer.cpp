#include "multi_struct_inferer.h"

MultiStructInferer::MultiStructInferer(const std::string sEngineFilePath)
    : m_inputDims({ 5, 3, 256, 256 })
    , m_outputEDDims({ 1, 4, 256, 256 })
    , m_outputESDims({ 1, 2, 256, 256 })
    , m_edClasses(4)
    , m_esClasses(2)
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
    m_outputEDMaskIdx = m_engine->getBindingIndex("ed");
    m_outputESMaskIdx = m_engine->getBindingIndex("es");

    m_context->setBindingDimensions(m_inputImageIdx, m_inputDims);
}

MultiStructInferer::~MultiStructInferer()
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

int MultiStructInferer::doInference(cv::Mat& src, std::vector<cv::Mat>& vMasks)
{
    return 0;
}

int MultiStructInferer::doInference(std::vector<cv::Mat>& imgs, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
    std::vector<cv::Mat> vMasks;
    doSingleInfer(imgs, vMasks);

    cv::Mat ivsMask = vMasks[0];
    cv::Mat pwMask = vMasks[1];
    cv::Mat aortaMask = vMasks[2];
    cv::Mat laMask = vMasks[3];

    std::vector<cv::Mat> vEdMasks = std::vector<cv::Mat>({ ivsMask, pwMask });

    ImageProcess::ivsAndPWPostProcess(*imgs.begin(), vEdMasks, values, resultPics);
    ImageProcess::ladPostProcess(imgs.back(), laMask, aortaMask, values, resultPics);
    ImageProcess::asdAndSJDAPostProcess(*imgs.begin(), aortaMask, values, resultPics);
    ImageProcess::aadPostProcess(*imgs.begin(), aortaMask, values, resultPics);
    return 1;
}

int MultiStructInferer::doInference(std::vector<cv::Mat>& imgs, std::vector<cv::Mat>& vMasks)
{
    return 0;
}

int MultiStructInferer::doSingleInfer(std::vector<cv::Mat>& imgs, std::vector<cv::Mat>& vMasks)
{
    void* inputMem{ nullptr };
    void* outputEDMem{ nullptr };
    void* outputESMem{ nullptr };
    size_t inputSize = getMemorySize(m_inputDims, sizeof(float));
    size_t outputEDSize = getMemorySize(m_outputEDDims, sizeof(float));
    size_t outputESSize = getMemorySize(m_outputESDims, sizeof(float));

    if (cudaMalloc(&inputMem, inputSize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: input cuda memory allocation failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }
    if (cudaMalloc(&outputEDMem, outputEDSize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: output cuda memory allocation failed, size = " << outputEDSize << " bytes" << std::endl;
        return 0;
    }
    if (cudaMalloc(&outputESMem, outputESSize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: output cuda memory allocation failed, size = " << outputESSize << " bytes" << std::endl;
        return 0;
    }

    cudaStream_t stream;  // 使用cudaStreamCreate函数创建CUDA流，用于在GPU上执行异步操作。
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return 0;
    }

    float* blob = new float[getElementNum(m_inputDims)];
    float* outputED = new float[getElementNum(m_outputEDDims)];
    float* outputES = new float[getElementNum(m_outputESDims)];
    preprocess(imgs, blob);

    if (cudaMemcpyAsync(inputMem, blob, inputSize, cudaMemcpyHostToDevice, stream) != cudaSuccess)  // 使用cudaMemcpyAsync函数将blob中的数据异步拷贝到GPU的输入内存inputMem中。
    {
        logger::gLogError << "ERROR: CUDA memory copy of input failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }

    // Run TensorRT inference
    void* bindings[] = { inputMem, outputEDMem, outputESMem };
    bool status = m_context->enqueueV2(bindings, stream, nullptr);  // 执行推理
    if (!status)
    {
        logger::gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return 0;
    }

    if (cudaMemcpyAsync(outputED, outputEDMem, outputEDSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)  // 使用cudaMemcpyAsync函数将GPU的输出内存outputMem中的数据异步拷贝到主机内存output中。
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << outputEDSize << " bytes" << std::endl;
        return 0;
    }

    if (cudaMemcpyAsync(outputES, outputESMem, outputESSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)  // 使用cudaMemcpyAsync函数将GPU的输出内存outputMem中的数据异步拷贝到主机内存output中。
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << outputESSize << " bytes" << std::endl;
        return 0;
    }

    cv::Mat edMask, esMask;
    vMasks = postProcess(outputED, outputES);

    delete[] blob;
    delete[] outputED;
    delete[] outputES;

    cudaFree(inputMem);
    cudaFree(outputEDMem);
    cudaFree(outputESMem);
    cudaStreamDestroy(stream);
    return 1;
}

cv::Mat MultiStructInferer::preprocessSingleImage(cv::Mat& src)
{
    cv::Mat dst;
    src.copyTo(dst);

    cv::Size inputSize(m_inputDims.d[3], m_inputDims.d[2]);  //size(width,height)

    cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);
    cv::resize(dst, dst, inputSize);
    dst.convertTo(dst, CV_32FC3);

    return dst;
}

int MultiStructInferer::preprocess(std::vector<cv::Mat>& imgs, float* blob)
{
    if (blob == nullptr)
        blob = new float[getElementNum(m_inputDims)];

    int counter = 0;
    for (auto& img : imgs)
    {
        cv::Mat dst = preprocessSingleImage(img);
        int channels = dst.channels();
        int rows = dst.rows;  // cols == width == Point.x;   rows == heigh == Point.y
        int cols = dst.cols;

        for (int c = 0; c < channels; c++)
        {
            for (int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++)
                {
                    blob[counter * channels * rows * cols + c * rows * cols + row * cols + col] = (dst.at<cv::Vec3f>(row, col)[c] / 255.0f - m_means[c]) / m_stds[c];
                }
            }
        }

        ++counter;
    }
    return 1;
}

std::vector<cv::Mat> MultiStructInferer::postProcess(float* outputED, float* outputES)
{
    cv::Size outputSize(m_outputEDDims.d[2], m_outputEDDims.d[3]);
    int height = outputSize.height;
    int width = outputSize.width;

    cv::Mat ivsMask, pwMask, aortaMask, laMask;
    ivsMask = cv::Mat::zeros(outputSize, CV_8UC1);
    pwMask = cv::Mat::zeros(outputSize, CV_8UC1);
    aortaMask = cv::Mat::zeros(outputSize, CV_8UC1);
    laMask = cv::Mat::zeros(outputSize, CV_8UC1);

    // Iterate through the mask data and assign pixel values to the resultMat

    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            float maxProb = 0.0f;
            int maxProbIdx = 0;
            for (int c = 0; c < m_edClasses; c++)
            {
                float currProb = outputED[c * height * width + height * h + w];
                if (currProb > maxProb)
                {
                    maxProb = currProb;
                    maxProbIdx = c;
                }
            }

            if (maxProbIdx == 1)
                ivsMask.at<uchar>(h, w) = 255;
            else if (maxProbIdx == 2)
                pwMask.at<uchar>(h, w) = 255;
            else if (maxProbIdx == 3)
                aortaMask.at<uchar>(h, w) = 255;
        }
    }

    //cv::imshow("ivs", ivsMask);
    //cv::imshow("pw", pwMask);
    //cv::imshow("aorta", aortaMask);
    //cv::waitKey(0);

    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            float maxProb = 0.0f;
            int maxProbIdx = 0;
            for (int c = 0; c < m_esClasses; c++)
            {
                float currProb = outputES[c * height * width + height * h + w];
                if (currProb > maxProb)
                {
                    maxProb = currProb;
                    maxProbIdx = c;
                }
            }

            if (maxProbIdx == 1)
                laMask.at<uchar>(h, w) = 255;
        }
    }

    return std::vector<cv::Mat>({ ivsMask, pwMask, aortaMask, laMask });
}
