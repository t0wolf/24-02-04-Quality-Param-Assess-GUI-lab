#include "ef_video_inferer.h"

EFVideoInferer::EFVideoInferer(std::string& sEngineFilePath)
	:SegmentInferBase(sEngineFilePath)
{   
    // {1, 3, 32, 112, 112}
    m_inputDims.nbDims = 5;
	m_inputDims.d[0] = 1;
    m_inputDims.d[1] = 3;
    m_inputDims.d[2] = 32;
    m_inputDims.d[3] = 112;
    m_inputDims.d[4] = 112;

    m_outputDims.nbDims = 2;
	m_outputDims.d[0] = 1;
    m_outputDims.d[1] = 1;
	m_classes = 1;
}

EFVideoInferer::~EFVideoInferer()
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

int EFVideoInferer::doInference(std::vector<cv::Mat>& video, std::vector<float>& predScores)
{
    std::vector<std::vector<cv::Mat>> clipedVideos;
    float score;
    clipVideo(video, clipedVideos);
    for (auto& oneClip : clipedVideos) 
    {
  
        //for test
        //for (auto& frame : oneClip) 
        //{
        //    cv::imshow("test", frame);
        //    cv::waitKey(0);
        //}

        doSingleInfer(oneClip, score);
        predScores.push_back(score);
    }

	return 0;
}

int EFVideoInferer::clipVideo(std::vector<cv::Mat>& srcVideo, std::vector<std::vector<cv::Mat>>& clipedVideos)
{
    std::vector<cv::Mat> dstVideo(srcVideo);
    const int length = 32;
    const int period = 2;
    int frameCount = dstVideo.size();
    if (frameCount < length * period) // 如果小于64帧需要标准化之后再填充
    {
        std::cout << "输入的视频帧数少于64帧" << std::endl;
    }

    int start = frameCount - (length - 1) * period;

    for (int s = 0; s < start; s++) // 每隔两帧取一帧，共取32帧，同时保证不超过视频长度
    {
        std::vector<cv::Mat> oneClip;
        for (int p = 0; p < length; p++) 
        {
            // for test
            //int idx = s + period * p;
            //std::string text = std::to_string(idx);
            //cv::putText(dstVideo[s + period * p], text, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 2);
            //cv::imshow("test", dstVideo[s + period * p]);
            //cv::waitKey(0);

            oneClip.push_back(dstVideo[s + period * p]);
        }
        clipedVideos.push_back(oneClip);
    }

    return 0;
}

int EFVideoInferer::doSingleInfer(std::vector<cv::Mat>& video, float& score)
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

    postProcess(output, score);

    delete[] blob;
    delete[] output;
    cudaFree(inputMem);
    cudaFree(outputMem);
    cudaStreamDestroy(stream);

	return 0;
}


int EFVideoInferer::preprocess(std::vector<cv::Mat>& video, float* blob)
{   
    std::vector<cv::Mat> dstVideo(video);
    cv::Size inputSize(m_inputDims.d[4], m_inputDims.d[3]);  //size(width,height)

    for (auto& frame : dstVideo) 
    {
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        cv::resize(frame, frame, inputSize);
        frame.convertTo(frame, CV_32FC3);
    }

    blobFromImage(dstVideo, blob);
	return 0;
}

int EFVideoInferer::blobFromImage(std::vector<cv::Mat>& video, float* blob) 
{
    cv::Mat src = video[0];

    int channels = src.channels();
    int rows = src.rows;  // cols == width == Point.x;   rows == heigh == Point.y
    int cols = src.cols;
    int frameCount = video.size();
    float mean[3] = { 32.55479f, 32.68759f, 32.989567f };
    float std[3] = { 49.92539f, 49.93201f, 50.176304f };

    // 存进去的顺序需要和输入的维度顺序一致  {1， 3， 32， 112， 112}

    //for (int f = 0; f < frameCount; f++)  // int f = 0; f < frameCount; f++
    //{
    //    for (int c = 0; c < channels; c++)
    //    {
    //        for (int row = 0; row < rows; row++)
    //        {
    //            for (int col = 0; col < cols; col++)
    //            {
    //                //float pixel = video[f].at<cv::Vec3f>(row, col)[c];
    //                //float normalizedPixel = (video[f].at<cv::Vec3f>(row, col)[c] - mean[c]) / std[c];
    //                blob[f * channels * rows * cols + c * rows * cols + row * cols + col] = (video[f].at<cv::Vec3f>(row, col)[c] - mean[c]) / std[c];
    //            }
    //        }
    //    }
    //        
    //}

    for (int c = 0; c < channels; c++)  // int f = 0; f < frameCount; f++
    {
        for (int f = 0; f < frameCount; f++)
        {
            for (int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++)
                {
                    blob[c * frameCount * rows * cols + f * rows * cols + row * cols + col] = (video[f].at<cv::Vec3f>(row, col)[c] - mean[c]) / std[c];
                }
            }
        }

    }
	return 0;
}

int EFVideoInferer::postProcess(float* output, float& score)
{
    score = output[0];
    //size_t size = sizeof(output);
 	return 0;
}