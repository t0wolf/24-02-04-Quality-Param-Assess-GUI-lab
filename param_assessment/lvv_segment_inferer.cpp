#include "lvv_segment_inferer.h"

LVVSegmentInferer::LVVSegmentInferer(std::string& sEngineFilePath)
	: SegmentInferBase(sEngineFilePath)
{	
	m_inputDims = { 3, 3, 256, 256 };  // frame, c, h, w
	m_outputDims = { 1, 2, 256, 256 };
	m_classes = 2;
}

LVVSegmentInferer::~LVVSegmentInferer()
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

int LVVSegmentInferer::doInference(std::vector<cv::Mat>& video, std::vector<cv::Mat>& vMasks)
{
    doSingleInfer(video, vMasks);
    return 0;
}

int LVVSegmentInferer::preprocess(std::vector<cv::Mat>& video, float* blob)
{
    std::vector<cv::Mat> dstVideo(video);  // pred1, pred2, current
    cv::Size inputSize(m_inputDims.d[3], m_inputDims.d[2]);  //size(width,height)

    for (auto& frame : dstVideo)
    {
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        cv::resize(frame, frame, inputSize);
        frame.convertTo(frame, CV_32FC3);
    }

    blobFromImage(dstVideo, blob);
    return 1;

}

int LVVSegmentInferer::blobFromImage(std::vector<cv::Mat>& video, float* blob)
{
    cv::Mat src = video[0];

    int channels = src.channels();
    int rows = src.rows;  // cols == width == Point.x;   rows == heigh == Point.y
    int cols = src.cols;
    int frameCount = video.size();

    // 存进去的顺序需要和输入的维度顺序一致  { 3， 3， 256， 256}

    for (int f = 0; f < frameCount; f++)  // int f = 0; f < frameCount; f++
    {
        for (int c = 0; c < channels; c++)
        {
            for (int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++)
                {
                    blob[f * channels * rows * cols + c * rows * cols + row * cols + col] = (video[f].at<cv::Vec3f>(row, col)[c]) / 255.0f;
                }
            }
        }

    }
    return 0;
}

int LVVSegmentInferer::doSingleInfer(std::vector<cv::Mat>& video, std::vector<cv::Mat>& vMasks)
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
    preprocess(video, blob);

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

int LVVSegmentInferer::postProcess(float* output, std::vector<cv::Mat>& vMasks)
{
    vMasks.clear();
    cv::Size outputSize(m_outputDims.d[3], m_outputDims.d[2]);
    int height = outputSize.height;  // m_outputDims.d[2]
    int width = outputSize.width;  // m_outputDims.d[3]

    cv::Mat lvvMask = cv::Mat::zeros(outputSize, CV_8UC1);

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
                lvvMask.at<uchar>(h, w) = 255;
        }
    }

    //cv::imshow("test", lvvMask);
    //cv::waitKey(0);

    vMasks = { lvvMask };

    return 1;
}

