#include "OutPaintInferer.h"

OutPaintInferer::OutPaintInferer(std::string& sEngineFilePath)
    :m_inputDims({ 5, 3, 192, 192 })  // {1, 5, 3, 256, 256}
    , m_outputDims({ 5, 3, 192, 192 })
{

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

    m_inputImageIdx = m_engine->getBindingIndex("input");
    m_outputIdx = m_engine->getBindingIndex("out");

    //m_context->setBindingDimensions(m_inputImageIdx, m_inputDims);
    //m_context->setBindingDimensions(m_outputEdIdx, m_outputEdDims);
    //m_context->setBindingDimensions(m_outputEdIdx, m_outputEsDims);
}

OutPaintInferer::~OutPaintInferer()
{
    if (m_context != nullptr)
    {
        delete m_context;
        m_context = nullptr;
    }
    if (m_engine != nullptr)
    {
        delete m_engine;
        m_engine = nullptr;
    }

}

int OutPaintInferer::doInference(std::vector<cv::Mat>& video, std::vector<cv::Mat>& outpaint_video)
{
    doSingleInfer(video, outpaint_video);

    return 0;
}

int OutPaintInferer::doSingleInfer(std::vector<cv::Mat>& video, std::vector<cv::Mat>& outpaint_video)
{
    void* inputMem{ nullptr };  // GPU中的内存
    void* outputEdMem{ nullptr };
    size_t inputSize = getMemorySize(m_inputDims, sizeof(float));
    size_t outputEdSize = getMemorySize(m_outputDims, sizeof(float));


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
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return 0;
    }

    float* blob = new float[getElementNum(m_inputDims)];  // CPU中的内存
    float* output = new float[getElementNum(m_outputDims)];

    preprocess(video, blob);

    if (cudaMemcpyAsync(inputMem, blob, inputSize, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of input failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }

    // Run TensorRT inference
    void* bindings[] = { inputMem, outputEdMem };
    bool status = m_context->enqueueV2(bindings, stream, nullptr);
    if (!status)
    {
        logger::gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return 0;
    }


    if (cudaMemcpyAsync(output, outputEdMem, outputEdSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << outputEdSize << " bytes" << std::endl;
        return 0;
    }


    postProcess(output, outpaint_video);
    cv::Size originalImageSize = video[0].size();
    for (int i = 0; i < outpaint_video.size(); i++)
    {
        cv::resize(outpaint_video[i], outpaint_video[i], originalImageSize);



    }
    delete[] blob;
    delete[] output;
    cudaFree(inputMem);
    cudaFree(outputEdMem);
    cudaStreamDestroy(stream);
    return 1;
    return 1;
}

int OutPaintInferer::preprocess(std::vector<cv::Mat>& video, float* blob)
{
    std::vector<cv::Mat> dstVideo(video);
    //cv::Mat originImg = video.back().clone();
    //cv::imshow("test", originImg);
    //cv::waitKey(0);

    cv::Size inputSize(m_inputDims.d[3], m_inputDims.d[2]);  //size(width,height)

    //for (auto& frame : dstVideo)
    //{
    //    cv::Mat dstFrame;
    //    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    //    cv::resize(frame, frame, inputSize);
    //    //cv::imshow("test2", frame);
    //    //cv::waitKey(0);
    //    frame.convertTo(frame, CV_32FC3);

    //}
    for (auto& frame : dstVideo)
    {
        cv::resize(frame, frame, inputSize);
        //frame.convertTo(frame, CV_32FC3); // 转为浮点

        // 将右半部分填充为(182,182,182)的值
        int half_width = frame.cols / 2;
        cv::Rect rightRect(half_width, 0, frame.cols - half_width, frame.rows);
        frame(rightRect).setTo(cv::Scalar(255.0f, 255.0f, 255.0f));
        //cv::imshow("test2", frame);
        //cv::waitKey(0);

    }
    //cv::Mat originImg2 = dstVideo.back().clone();


    blobFromImage(dstVideo, blob);
    //cv::imshow("test2", dstVideo[0]);
    //cv::waitKey(0);
    return 1;
}

//int OutPaintInferer::blobFromImage(std::vector<cv::Mat>& video, float* blob)
//{
//    cv::Mat src = video[0];
//    int channels = src.channels();
//    int rows = src.rows;  // cols == width == Point.x;   rows == heigh == Point.y
//    int cols = src.cols;
//    int frameCount = video.size();
//
//    float mean[3] = { 0.485f, 0.456f, 0.406f };
//    float std[3] = { 0.229f, 0.224f, 0.225f };
//
//    for (int f = 0; f < frameCount; f++)  // {1， 5， 3， 256， 256}
//    {
//        for (int c = 0; c < channels; c++)
//        {
//            for (int row = 0; row < rows; row++)
//            {
//                for (int col = 0; col < cols; col++)
//                {
//                    blob[f * channels * rows * cols + c * rows * cols + row * cols + col] = (video[f].at<cv::Vec3f>(row, col)[c] / 255.0f - mean[c]) / std[c];
//                }
//            }
//        }
//    }
//
//    return 1;
//}
int OutPaintInferer::blobFromImage(std::vector<cv::Mat>& video, float* blob)
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
                    blob[f * channels * rows * cols + c * rows * cols + row * cols + col] = (video[f].at<cv::Vec3b>(row, col)[c] / 255.0f - mean[c]) / std[c];
                }
            }
        }
    }

    return 1;
}
//    cv::Mat src = video[0];
//    int channels = src.channels();
//    int rows = src.rows;
//    int cols = src.cols;
//    int frameCount = static_cast<int>(video.size());
//    std::cout << "video size: " << video.size() << std::endl;
//    for (int i = 0; i < video.size(); i++) {
//        std::cout << "Frame " << i << ": " << video[i].rows << " x " << video[i].cols
//            << ", type=" << video[i].type() << std::endl;
//    }
//    // 不需要再进行归一化操作，直接赋值
//    for (int f = 0; f < frameCount; f++)
//    {
//        for (int c = 0; c < channels; c++)
//        {
//            for (int row = 0; row < rows; row++)
//            {
//                for (int col = 0; col < cols; col++)
//                {
//                    blob[f * channels * rows * cols
//                        + c * rows * cols
//                        + row * cols
//                        + col] = video[f].at<cv::Vec3b>(row, col)[c];
//                }
//            }
//        }
//    }
//
//    return 1;
//}
int OutPaintInferer::postProcess(float* outputEd,std::vector<cv::Mat>& outpaint_video)
{
    // 假设 m_outputDims = {T, C, H, W}
    int T = m_outputDims.d[0];
    int C = m_outputDims.d[1];
    int H = m_outputDims.d[2];
    int W = m_outputDims.d[3];

    // 仅处理第0帧数据
    int i = 0;

    //float mean[3] = { 0.5f, 0.5f, 0.5f };
    float std[3] = { 0.35f, 0.35f, 0.35f };
    float mean[3] = { 0.485f, 0.456f, 0.406f };
    //float std[3] = { 0.229f, 0.224f, 0.225f };
    // 创建存储图像的 Mat (H,W,C=3)浮点图像
    cv::Mat pre_img_mat(H, W, CV_32FC3);
    for (int i = 0; i < T; i++) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                cv::Vec3f pixel;
                for (int c_idx = 0; c_idx < C; ++c_idx) {
                    float val_pre = outputEd[i * C * H * W + c_idx * H * W + h * W + w];

                    // 反归一化 val_pre = val_pre * std[c_idx] + mean[c_idx]
                    val_pre = val_pre * std[c_idx] + mean[c_idx];

                    // 裁剪到 [0, 1]
                    val_pre = std::max(0.0f, std::min(1.0f, val_pre));
                    pixel[c_idx] = val_pre;
                }
                pre_img_mat.at<cv::Vec3f>(h, w) = pixel;
            }
        }
        cv::Mat save_img;
        pre_img_mat.convertTo(save_img, CV_8UC3, 255.0); // 转换为8位图像

        // 构造保存路径和文件名：例如存放在 "./output_frames" 文件夹中
        // 文件名为 "0.png", "1.png", "2.png", ...
        //std::string folder_path = "D:/test/outpaint";
        //std::string file_name = std::to_string(i) + "111.png";
        //std::string full_path = folder_path + "/" + file_name;

        //// 将图像写入文件
        //cv::imwrite(full_path, save_img);
        //cv::resize(pre_img_mat, pre_img_mat, cv::Size(256,256));
        //cv::imshow("test2", save_img);
        //cv::waitKey(0);
        outpaint_video.push_back(save_img);
    }


    // 如果需要与 Python 中RGB显示一致，这里做 BGR->RGB 转换
    //cv::Mat pre_img_rgb;
    //cv::cvtColor(pre_img_mat, pre_img_rgb, cv::COLOR_BGR2RGB);

    //// 将浮点图像(0~1)转换为8位图像(0~255)
    //cv::Mat display_img;
    //pre_img_rgb.convertTo(display_img, CV_8UC3, 255);



    //cv::imshow("test2", pre_img_mat);
    //cv::waitKey(0);




    return 0;
}




