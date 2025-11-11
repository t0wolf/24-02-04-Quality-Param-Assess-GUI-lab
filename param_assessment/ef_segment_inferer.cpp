#include "ef_segment_inferer.h"

EFSegmentInferer::EFSegmentInferer(std::string& sEngineFilePath) 
	:SegmentInferBase(sEngineFilePath)
{
	m_inputDims = { 1, 3, 112, 112 };  // {height, width}
	m_outputDims = { 1, 1, 112, 112 };
	m_classes = 1;
}

EFSegmentInferer::~EFSegmentInferer() 
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

int EFSegmentInferer::doInference(std::vector<cv::Mat>& video, std::vector<int>& framePixels, std::vector<cv::Mat>& frameMasks)
{   
    int lvPixels;
    
    for (auto& frame : video) 
    {   
        cv::Mat lvMask;
        //cv::imshow("test", frame);
        //cv::waitKey(0);

        doSingleInfer(frame, lvPixels, lvMask);
        framePixels.push_back(lvPixels);
        frameMasks.push_back(lvMask);
    }

	return 1;
}

int EFSegmentInferer::doSingleInfer(cv::Mat& src, int& lvPixels, cv::Mat& lvMask)
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

    postProcess(output, lvPixels, lvMask); 

    delete[] blob;
    delete[] output;
    cudaFree(inputMem);
    cudaFree(outputMem);
    cudaStreamDestroy(stream);

    return 1;
}

int EFSegmentInferer::preprocess(cv::Mat& src, float* blob) 
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

int EFSegmentInferer::blobFromImage(cv::Mat& src, float* blob) 
{
	int channels = src.channels();
    int rows = src.rows;  // cols == width == Point.x;   rows == heigh == Point.y
    int cols = src.cols;
    float mean[3] = { 32.55479f, 32.68759f, 32.989567f };
    float std[3] = { 49.92539f, 49.93201f, 50.176304f };

    for (int c = 0; c < channels; c++)
    {
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                blob[c * rows * cols + row * cols + col] = (src.at<cv::Vec3f>(row, col)[c] - mean[c]) / std[c];  
            }
        }
    }
    return 1;
}

int EFSegmentInferer::postProcess(float* output, int& laPixels, cv::Mat& lvMask)
{   
    cv::Size outputSize(m_outputDims.d[3], m_outputDims.d[2]);
    int height = outputSize.height;  // m_outputDims.d[2]
    int width = outputSize.width;  // m_outputDims.d[3]

    lvMask = cv::Mat::zeros(outputSize, CV_8UC1);

    // Iterate through the mask data and assign pixel values to the resultMat

    laPixels = 0;
    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            float prob = 0.0f;
            for (int c = 0; c < m_classes; c++)
            {
                float currProb = output[c * height * width + height * h + w];
                if (currProb > prob)
                {   
                    laPixels++;
                    lvMask.at<uchar>(h, w) = 255;
                }
            }
        }
    }

    return 1;
}