#include "segment_infer_base.h"

SegmentInferBase::SegmentInferBase(const std::string& sEngineFilePath)  // 构造函数
    : m_inputDims({1, 3, 512, 512})
    , m_outputDims({1, 3, 512, 512})
    , m_classes(3)
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


    //SegLogger logger;
    auto log = logger::gLogger;
    log.setReportableSeverity(nvinfer1::ILogger::Severity::kVERBOSE);
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(log.getTRTLogger());
    m_engine = runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
    assert(m_engine != nullptr);

    m_context = m_engine->createExecutionContext();
    assert(m_context != nullptr);

    m_inputImageIdx = m_engine->getBindingIndex("images");
    m_outputMaskIdx = m_engine->getBindingIndex("output");

    //m_context->setBindingDimensions(m_inputImageIdx, m_inputDims);
}

SegmentInferBase::~SegmentInferBase()  // 析构函数
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

int SegmentInferBase::doInference(cv::Mat& src, cv::Mat& mask)  // 执行推理
{
    doSingleInfer(src, mask);
    return 1;
}

int SegmentInferBase::doInference(cv::Mat& src, std::vector<cv::Mat>& vMasks)
{
    doSingleInfer(src, vMasks);
    return 1;
}

int SegmentInferBase::preprocess(cv::Mat& src, float* blob)  // 图像预处理，并存入内存
{
    cv::Mat dst;
    src.copyTo(dst);

    cv::Size inputSize(m_inputDims.d[3], m_inputDims.d[2]);  //size(width,height)

    cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);
    cv::resize(dst, dst, inputSize);
    dst.convertTo(dst, CV_32FC3);
    blobFromImage(dst, blob);
    return 1;
}

int SegmentInferBase::blobFromImage(cv::Mat& src, float* blob)  // 遍历图像，将每一个像素存到一维数组当中(每一个c，按照其上一行一行的顺序存进去)
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
                blob[c * rows * cols + row * cols + col] = src.at<cv::Vec3f>(row, col)[c] / 255.0f;
            }
        }
    }
    return 1;
}

int SegmentInferBase::doSingleInfer(cv::Mat& src, cv::Mat& mask)
{
    // 分配内存
    void* inputMem{ nullptr };
    void* outputMem{ nullptr };
    size_t inputSize = getMemorySize(m_inputDims, sizeof(float));
    size_t outputSize = getMemorySize(m_outputDims, sizeof(float));

    if (cudaMalloc(&inputMem, inputSize) != cudaSuccess)  // 使用cudaMalloc函数在GPU上为输入数据分配内存，将分配的内存地址存储在inputMem中。
    {
        logger::gLogError << "ERROR: input cuda memory allocation failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }
    if (cudaMalloc(&outputMem, outputSize) != cudaSuccess)  // 使用cudaMalloc函数在GPU上为输出数据分配内存，将分配的内存地址存储在outputMem中。
    {
        logger::gLogError << "ERROR: output cuda memory allocation failed, size = " << outputSize << " bytes" << std::endl;
        return 0;
    }

    cudaStream_t stream;  // 使用cudaStreamCreate函数创建CUDA流，用于在GPU上执行异步操作。
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return 0;
    }

    float* blob = new float[getElementNum(m_inputDims)];  // 使用getElementNum函数计算输入数据的元素个数，并使用new操作符在堆上分配内存，将分配的内存地址存储在blob中。
    float* output = new float[getElementNum(m_outputDims)];  // 使用getElementNum函数计算输出数据的元素个数，并使用new操作符在堆上分配内存，将分配的内存地址存储在output中。
    preprocess(src, blob);  // 图像预处理，将预处理后的数据存储在blob中。

    if (cudaMemcpyAsync(inputMem, blob, inputSize, cudaMemcpyHostToDevice, stream) != cudaSuccess)  // 使用cudaMemcpyAsync函数将blob中的数据异步拷贝到GPU的输入内存inputMem中。
    {
        logger::gLogError << "ERROR: CUDA memory copy of input failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }

    // Run TensorRT inference
    void* bindings[] = { inputMem, outputMem };
    bool status = m_context->enqueueV2(bindings, stream, nullptr);  // 执行推理
    if (!status)
    {
        logger::gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return 0;
    }

    if (cudaMemcpyAsync(output, outputMem, outputSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)  // 使用cudaMemcpyAsync函数将GPU的输出内存outputMem中的数据异步拷贝到主机内存output中。
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << outputSize << " bytes" << std::endl;
        return 0;
    }

    postProcess(output, mask);  // 后处理，调用postProcess函数对输出数据进行后处理，并将结果存储在mask中。

    delete[] blob;
    delete[] output;
    cudaFree(inputMem);
    cudaFree(outputMem);
    cudaStreamDestroy(stream);
    return 1;
}

int SegmentInferBase::doSingleInfer(cv::Mat& src, std::vector<cv::Mat>& vMasks)
{
    void* inputMem{ nullptr };
    void* outputMem{ nullptr };
    size_t inputSize = getMemorySize(m_inputDims, sizeof(float));
    size_t outputSize = getMemorySize(m_outputDims, sizeof(float));

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

    float* blob = new float[getElementNum(m_inputDims)];
    float* output = new float[getElementNum(m_outputDims)];
    preprocess(src, blob);

    if (cudaMemcpyAsync(inputMem, blob, inputSize, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of input failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }

    // Run TensorRT inference
    void* bindings[] = { inputMem, outputMem };
    bool status = m_context->enqueueV2(bindings, stream, nullptr);
    if (!status)
    {
        logger::gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return 0;
    }

    if (cudaMemcpyAsync(output, outputMem, outputSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << outputSize << " bytes" << std::endl;
        return 0;
    }

    postProcess(output, vMasks);

    delete[] blob;
    delete[] output;
    cudaFree(inputMem);
    cudaFree(outputMem);
    cudaStreamDestroy(stream);
    return 1;
}

int SegmentInferBase::postProcess(float* output, cv::Mat& mask)  // 后处理，遍历输出数据的每个像素位置，找到概率最大的类别，并根据类别索引设置mask中的像素值。
                                                                 // 最终生成的mask表示图像的分割结果，像素值为255的区域表示属于目标类别，像素值为0的区域表示不属于目标类别。
{
    cv::Size outputSize(m_outputDims.d[2], m_outputDims.d[3]);
    int height = outputSize.height;
    int width = outputSize.width;

    if (mask.empty())
        mask = cv::Mat::zeros(outputSize, CV_32SC1);
    // Iterate through the mask data and assign pixel values to the resultMat
    
    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            float maxProb = 0.0f;
            int maxProbIdx = 0;
            for (int c = 0; c < m_classes; c++)
            {
                float currProb = output[c * height * width + height * h + w];  // 计算当前像素位置（h，w）处类别c的概率值。
                if (currProb > maxProb)
                {
                    maxProb = currProb;
                    maxProbIdx = c;
                }
            }

            mask.at<int>(h, w) = maxProbIdx == 1 ? 255 : 0;  // 根据最大概率对应的类别索引判断像素值，如果最大概率对应的类别索引为1，则将mask中位置（h，w）处的像素值设为255，否则设为0。
        }
    }

    return 1;
}

int SegmentInferBase::postProcess(float* output, std::vector<cv::Mat>& vMasks)
{
    vMasks.clear();
    cv::Size outputSize(m_outputDims.d[2], m_outputDims.d[3]);
    int height = outputSize.height;
    int width = outputSize.width;

    cv::Mat pwMask, ivsMask;
    pwMask = cv::Mat::zeros(outputSize, CV_32SC1);
    ivsMask = cv::Mat::zeros(outputSize, CV_32SC1);
    // Iterate through the mask data and assign pixel values to the resultMat

    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            float maxProb = 0.0f;
            int maxProbIdx = 0;
            for (int c = 0; c < m_classes; c++)
            {
                float currProb = output[c * height * width + height * h + w];
                if (currProb > maxProb)
                {
                    maxProb = currProb;
                    maxProbIdx = c;
                }
            }

            if (maxProbIdx == 1)
                ivsMask.at<int>(h, w) = 255;
            else if (maxProbIdx == 2)
                pwMask.at<int>(h, w) = 255;
        }
    }

    vMasks = { ivsMask, pwMask };

    return 1;
}
