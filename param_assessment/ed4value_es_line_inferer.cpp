#include "ed4value_es_line_inferer.h"

DetachFourLineInferer::DetachFourLineInferer(std::string& sEngineFilePath)
    :m_inputDims({ 5, 3, 256, 256 })  // {1, 5, 3, 256, 256}
    , m_outputEdDims({ 1, 4, 256, 256 })
    , m_outputEsDims({ 1, 1, 256, 256 })
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
    m_outputEdIdx = m_engine->getBindingIndex("output_ed");
    m_outputEsIdx = m_engine->getBindingIndex("output_es");

    m_context->setBindingDimensions(m_inputImageIdx, m_inputDims);
    m_context->setBindingDimensions(m_outputEdIdx, m_outputEdDims);
    m_context->setBindingDimensions(m_outputEsIdx, m_outputEsDims);
}

DetachFourLineInferer::~DetachFourLineInferer()
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

int DetachFourLineInferer::doInference(std::vector<cv::Mat>& video, std::vector<std::vector<cv::Mat>>& vMasks)
{
    doSingleInfer(video, vMasks);

    return 0;
}

int DetachFourLineInferer::doSingleInfer(std::vector<cv::Mat>& video, std::vector<std::vector<cv::Mat>>& vMasks)
{
    void* inputMem{ nullptr };  // GPU中的内存
    void* outputEdMem{ nullptr };
    void* outputEsMem{ nullptr };
    size_t inputSize = getMemorySize(m_inputDims, sizeof(float));
    size_t outputEdSize = getMemorySize(m_outputEdDims, sizeof(float));
    size_t outputEsSize = getMemorySize(m_outputEsDims, sizeof(float));

    if (cudaMalloc(&inputMem, inputSize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: input cuda memory allocation failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }
    if (cudaMalloc(&outputEdMem, outputEdSize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: output cuda memory allocation failed, size = " << outputEdSize << " bytes" << std::endl;
        return 0;
    }
    if (cudaMalloc(&outputEsMem, outputEsSize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: output cuda memory allocation failed, size = " << outputEsSize << " bytes" << std::endl;
        return 0;
    }

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return 0;
    }

    float* blob = new float[getElementNum(m_inputDims)];  // CPU中的内存
    float* outputEd = new float[getElementNum(m_outputEdDims)];
    float* outputEs = new float[getElementNum(m_outputEsDims)];

    preprocess(video, blob);

    if (cudaMemcpyAsync(inputMem, blob, inputSize, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of input failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }

    // Run TensorRT inference
    void* bindings[] = { inputMem, outputEdMem, outputEsMem };
    bool status = m_context->enqueueV2(bindings, stream, nullptr);
    if (!status)
    {
        logger::gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return 0;
    }

    if (cudaMemcpyAsync(outputEd, outputEdMem, outputEdSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << outputEdSize << " bytes" << std::endl;
        return 0;
    }
    if (cudaMemcpyAsync(outputEs, outputEsMem, outputEsSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << outputEsSize << " bytes" << std::endl;
        return 0;
    }

    postProcess(outputEd, outputEs, vMasks);

    delete[] blob;
    delete[] outputEd;
    delete[] outputEs;
    cudaFree(inputMem);
    cudaFree(outputEdMem);
    cudaFree(outputEsMem);
    cudaStreamDestroy(stream);
    return 1;
    return 1;
}

int DetachFourLineInferer::preprocess(std::vector<cv::Mat>& video, float* blob)
{
    std::vector<cv::Mat> dstVideo(video);
    //cv::Mat originImg = video.back().clone();
    //cv::imshow("test", originImg);
    //cv::waitKey(0);

    cv::Size inputSize(m_inputDims.d[3], m_inputDims.d[2]);  //size(width,height)

    for (auto& frame : dstVideo)
    {
        cv::Mat dstFrame;
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        cv::resize(frame, frame, inputSize);
        //cv::imshow("test2", frame);
        //cv::waitKey(0);
        frame.convertTo(frame, CV_32FC3);

    }

    //cv::Mat originImg2 = dstVideo.back().clone();
    //cv::imshow("test2", originImg2);
    //cv::waitKey(0);

    blobFromImage(dstVideo, blob);
    return 1;
}

int DetachFourLineInferer::blobFromImage(std::vector<cv::Mat>& video, float* blob)
{
    cv::Mat src = video[0];
    int channels = src.channels();
    int rows = src.rows;  // cols == width == Point.x;   rows == heigh == Point.y
    int cols = src.cols;
    int frameCount = video.size();

    float mean[3] = { 0.485f, 0.456f, 0.406f };
    float std[3] = { 0.229f, 0.224f, 0.225f };

    for (int f = 0; f < frameCount; f++)  // {1， 5， 3， 256， 256}
    {
        for (int c = 0; c < channels; c++)
        {
            for (int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++)
                {
                    blob[f * channels * rows * cols + c * rows * cols + row * cols + col] = (video[f].at<cv::Vec3f>(row, col)[c] / 255.0f - mean[c]) / std[c];
                }
            }
        }
    }

    return 1;
}

int DetachFourLineInferer::postProcess(float* outputEd, float* outputEs, std::vector<std::vector<cv::Mat>>& vMasks)
{
    vMasks.clear();
    cv::Size outputEdSize(m_outputEdDims.d[2], m_outputEdDims.d[3]);
    int height = outputEdSize.height;  // m_outputDims.d[2]  此处ed和es宽高一致
    int width = outputEdSize.width;  // m_outputDims.d[3]
    int numMaskEd = m_outputEdDims.d[1];  // { 1, 6, 256, 256 }
    int numMaskEs = m_outputEsDims.d[1];  // { 1, 1, 256, 256 }

    std::vector<cv::Mat> masksEd(numMaskEd);
    std::vector<cv::Mat> masksEs(numMaskEs);

    for (size_t i = 0; i < numMaskEd; i++)
    {
        masksEd[i] = cv::Mat::zeros(outputEdSize, CV_32FC1);
    }

    for (size_t i = 0; i < numMaskEs; i++)
    {
        masksEs[i] = cv::Mat::zeros(outputEdSize, CV_32FC1);
    }

    for (int num = 0; num < numMaskEd; num++)
    {
        //cv::Mat binaryMap;
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                float currentValue = outputEd[num * height * width + h * width + w];
                if (currentValue < 0.f)
                {
                    currentValue = 0.f;
                }
                masksEd[num].at<float>(h, w) = currentValue;
            }
        }
        //cv::threshold(masksEd[num], binaryMap, 150, 255, cv::THRESH_BINARY);
        //cv::imshow("test", binaryMap);
        //cv::waitKey(0);

    }

    for (int num = 0; num < numMaskEs; num++)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                float currentValue = outputEs[num * height * width + h * width + w];
                if (currentValue < 0.f)
                {
                    currentValue = 0.f;
                }
                masksEs[num].at<float>(h, w) = currentValue;
            }
        }
    }

    vMasks = { masksEd, masksEs };

    return 0;
}




