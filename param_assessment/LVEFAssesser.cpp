#include "LVEFAssesser.h"

LVEFAssesser::LVEFAssesser(std::string& sEnginePath)
	: SegmentInferBase(sEnginePath)
	, m_aggVideoLength(16)
{
	m_inputDims = { 8, 3, 320, 320 };  // {height, width}
	m_outputDims = { 1, 2, 320, 320 };
    m_outputMaskDims = { 1, 2, 320, 320 };
	m_classes = 2;

    m_inputA2CIdx = m_engine->getBindingIndex("a2c_video");
    m_inputA4CIdx = m_engine->getBindingIndex("a4c_video");
    m_outputA2CEDIdx = m_engine->getBindingIndex("a2c_ed_logit");
    m_outputA2CESIdx = m_engine->getBindingIndex("a2c_es_logit");
    m_outputA4CEDIdx = m_engine->getBindingIndex("a4c_ed_logit");
    m_outputA4CESIdx = m_engine->getBindingIndex("a4c_es_logit");
}

LVEFAssesser::~LVEFAssesser()
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

int LVEFAssesser::doInference(std::vector<cv::Mat>& a2cVideo, std::vector<cv::Mat>& a4cVideo, std::vector<std::vector<cv::Mat>>& vMasks)
{
    doSingleInfer(a2cVideo, a4cVideo, vMasks);

    return 0;
}

int LVEFAssesser::blobFromImage(std::vector<cv::Mat>& video, float* blob)
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

int LVEFAssesser::doSingleInfer(std::vector<cv::Mat>& a2cVideo, std::vector<cv::Mat>& a4cVideo, std::vector<std::vector<cv::Mat>>& vMasks)
{
    void* input2CMem{ nullptr };  // GPU中的内存
    void* input4CMem{ nullptr };
    void* output2CEdMem{ nullptr };
    void* output2CEsMem{ nullptr };
    void* output4CEdMem{ nullptr };
    void* output4CEsMem{ nullptr };

    size_t inputSize = getMemorySize(m_inputDims, sizeof(float));
    size_t outputMaskSize = getMemorySize(m_outputMaskDims, sizeof(float));

    if (cudaMalloc(&input2CMem, inputSize) != cudaSuccess ||
        cudaMalloc(&input4CMem, inputSize) != cudaSuccess ||
        cudaMalloc(&output2CEdMem, outputMaskSize) != cudaSuccess ||
        cudaMalloc(&output2CEsMem, outputMaskSize) != cudaSuccess ||
        cudaMalloc(&output4CEdMem, outputMaskSize) != cudaSuccess ||
        cudaMalloc(&output4CEsMem, outputMaskSize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory allocation failed." << std::endl;
        return 0;
    }

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return 0;
    }

    std::unique_ptr<float[]> blob2C(new float[getElementNum(m_inputDims)]);
    std::unique_ptr<float[]> blob4C(new float[getElementNum(m_inputDims)]);
    std::unique_ptr<float[]> output2CEd(new float[getElementNum(m_outputMaskDims)]);
    std::unique_ptr<float[]> output2CEs(new float[getElementNum(m_outputMaskDims)]);
    std::unique_ptr<float[]> output4CEd(new float[getElementNum(m_outputMaskDims)]);
    std::unique_ptr<float[]> output4CEs(new float[getElementNum(m_outputMaskDims)]);

    //float* blob2C = new float[getElementNum(m_inputDims)];  // CPU中的内存
    //float* blob4C = new float[getElementNum(m_inputDims)];
    //float* output2CEd = new float[getElementNum(m_outputMaskDims)];
    //float* output2CEs = new float[getElementNum(m_outputMaskDims)];
    //float* output4CEd = new float[getElementNum(m_outputMaskDims)];
    //float* output4CEs = new float[getElementNum(m_outputMaskDims)];

    preprocess(a2cVideo, blob2C.get());
    preprocess(a4cVideo, blob4C.get());

    if (cudaMemcpyAsync(input2CMem, blob2C.get(), inputSize, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(input4CMem, blob4C.get(), inputSize, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of input failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }

    // Run TensorRT inference
    void* bindings[] = { input2CMem, input4CMem, output2CEdMem, output2CEsMem, output4CEdMem, output4CEsMem };
    bool status = m_context->enqueueV2(bindings, stream, nullptr);
    if (!status)
    {
        logger::gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return 0;
    }

    if (cudaMemcpyAsync(output2CEd.get(), output2CEdMem, outputMaskSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(output2CEs.get(), output2CEsMem, outputMaskSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << outputMaskSize << " bytes" << std::endl;
        return 0;
    }

    if (cudaMemcpyAsync(output4CEd.get(), output4CEdMem, outputMaskSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(output4CEs.get(), output4CEsMem, outputMaskSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << outputMaskSize << " bytes" << std::endl;
        return 0;
    }

    std::vector<cv::Mat> v2CEdMasks, v2CEsMasks, v4CEdMasks, v4CEsMasks;
    postProcess(output2CEd.get(), v2CEdMasks);
    postProcess(output2CEs.get(), v2CEsMasks);
    postProcess(output4CEd.get(), v4CEdMasks);
    postProcess(output4CEs.get(), v4CEsMasks);

    vMasks = { v2CEdMasks, v2CEsMasks, v4CEdMasks, v4CEsMasks };

    //delete[] blob2C;
    //delete[] blob4C;
    //delete[] output2CEd;
    //delete[] output2CEs;
    //delete[] output4CEd;
    //delete[] output4CEs;

    cudaFree(input2CMem);
    cudaFree(input4CMem);
    cudaFree(output2CEdMem);
    cudaFree(output2CEsMem);
    cudaFree(output4CEdMem);
    cudaFree(output4CEsMem);

    cudaStreamDestroy(stream);

    return 1;
}

int LVEFAssesser::postProcess(float* outputMask, std::vector<cv::Mat>& vMasks)
{
    cv::Size outputSize(m_outputDims.d[2], m_outputDims.d[3]);
    int height = outputSize.height;
    int width = outputSize.width;
    cv::Mat mask;

    if (mask.empty())
        mask = cv::Mat::zeros(outputSize, CV_8UC1);
    // Iterate through the mask data and assign pixel values to the resultMat

    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            float maxProb = 0.0f;
            int maxProbIdx = 0;
            for (int c = 0; c < m_classes; c++)
            {
                float currProb = outputMask[c * height * width + height * h + w];  // 计算当前像素位置（h，w）处类别c的概率值。
                if (currProb > maxProb)
                {
                    maxProb = currProb;
                    maxProbIdx = c;
                }
            }

            mask.at<uint8_t>(h, w) = maxProbIdx == 1 ? 255 : 0;  // 根据最大概率对应的类别索引判断像素值，如果最大概率对应的类别索引为1，则将mask中位置（h，w）处的像素值设为255，否则设为0。
        }
    }

    //cv::imshow("lvef_mask", mask);
    //cv::waitKey(0);

    vMasks = std::vector<cv::Mat>{mask};
    return 1;
}

int LVEFAssesser::preprocess(std::vector<cv::Mat>& video, float* blob)
{
    std::vector<cv::Mat> dstVideo(video);

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

    blobFromImage(dstVideo, blob);
    return 1;
}
