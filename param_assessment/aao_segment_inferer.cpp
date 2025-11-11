#include "aao_segment_inferer.h"

AAOSegmentInferer::AAOSegmentInferer(std::string& sEngineFilePath)
	: SegmentInferBase(sEngineFilePath)
{
	m_classes = 2;
    m_outputDims = { 1, 2, 512, 512 };
}

AAOSegmentInferer::~AAOSegmentInferer()
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

int AAOSegmentInferer::doInference(cv::Mat& src, std::vector<cv::Mat>& vMasks)
{
    int ret = doSingleInfer(src, vMasks);
    return ret;
}

int AAOSegmentInferer::doSingleInfer(cv::Mat& src, std::vector<cv::Mat>& vMasks)
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
    //cv::imshow("test", src);
    //cv::waitKey(0);
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

    this->postProcess(output, vMasks);

    delete[] blob;
    delete[] output;
    cudaFree(inputMem);
    cudaFree(outputMem);
    cudaStreamDestroy(stream);
    return 1;
}

int AAOSegmentInferer::postProcess(float* output, std::vector<cv::Mat>& vMasks)
{
    vMasks.clear();
    cv::Size outputSize(m_outputDims.d[2], m_outputDims.d[3]);
    int height = outputSize.height;  // m_outputDims.d[2]
    int width = outputSize.width;  // m_outputDims.d[3]

    cv::Mat aaoMask = cv::Mat::zeros(outputSize, CV_8UC1);

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
                aaoMask.at<uchar>(h, w) = 255;
        }
    }

    //cv::imshow("test", aaoMask);
    //cv::waitKey(0);

    vMasks = { aaoMask };

	return 1;
}
