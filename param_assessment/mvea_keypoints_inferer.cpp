#include "mvea_keypoints_inferer.h"

MVEAKeypointsInferer::MVEAKeypointsInferer(std::string& sEnginePath)
	:AVADKeypointsInferer(sEnginePath)
{
	m_inputDims = { 1, 3, 256, 192 };  // {height, width}
	m_outputDims = { 1, 2, 64, 48 };
	m_classes = 2;
}

MVEAKeypointsInferer::~MVEAKeypointsInferer()
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

int MVEAKeypointsInferer::doInference(cv::Mat& src, std::vector<cv::Point>& eaPoints)
{
    int ret = doSingleInfer(src, eaPoints);
    return ret;
}

int MVEAKeypointsInferer::doSingleInfer(cv::Mat& src, std::vector<cv::Point>& eaPoints)
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

    postProcess(output, src, eaPoints);  // 后处理，调用postProcess函数对输出数据进行后处理

    delete[] blob;
    delete[] output;
    cudaFree(inputMem);
    cudaFree(outputMem);
    cudaStreamDestroy(stream);
    return 1;
}

int MVEAKeypointsInferer::preprocess(cv::Mat& src, float* blob)
{
    cv::Mat dst;
    std::pair<cv::Mat, cv::Mat> transPair;
    src.copyTo(dst);

    int fixedSize[2] = { m_inputDims.d[2], m_inputDims.d[3] };  // {h, w}
    //cv::Size inputSize(m_inputDims.d[3], m_inputDims.d[2]);   //size(width,height)
    cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);
    //cv::resize(dst, dst, inputSize);

    transPair = ImageProcess::affineTransform(dst, fixedSize, 1.25f);

    cv::Mat resizedImg = transPair.first;
    m_invertTrans = transPair.second;

    //for test
    //cv::imshow("test", resizedImg);
    //cv::waitKey(0);

    resizedImg.convertTo(resizedImg, CV_32FC3);  // 将图像转换为三通道浮点数类型;
    // 彩色图像（RGB 或 BGR 格式），则通常情况下它的通道类型可能是 CV_8UC3，即每个像素由 8 位无符号整数表示，具有三个通道。

    blobFromImage(resizedImg, blob);
    return 1;
}

int MVEAKeypointsInferer::postProcess(float* output, cv::Mat& src, std::vector<cv::Point>& eaPoints)
{
    int height = m_outputDims.d[2];  // 
    int width = m_outputDims.d[3];  // 
    std::vector<cv::Point2f> MaxPoints;
    for (int c = 0; c < m_classes; c++)
    {
        float maxProb = 0.0f;
        cv::Point maxPoint;
        cv::Point2f floatMaxPoint;

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                float currProb = output[c * height * width + width * h + w];
                if (currProb > maxProb)
                {
                    maxProb = currProb;
                    maxPoint.x = w;
                    maxPoint.y = h;
                }
            }
        }

        // 源代码此处有筛选该两点最大概率是否大于0，如果不大于0则将该点坐标置于（0，0）的处理(但是maxProb=0.0)

        if ((maxPoint.x > 1 && maxPoint.x < (width - 1)) && (maxPoint.y > 1 && maxPoint.y < (height - 1)))
            // if ((1 < maxPoint.x < (width - 1)) && (1 < maxPoint.y < (height - 1)))  // 对xy坐标进行修正
        {
            float rProb = output[c * height * width + width * maxPoint.y + maxPoint.x + 1];  // 该点相邻右侧点的概率
            float lProb = output[c * height * width + width * maxPoint.y + maxPoint.x - 1];  // 该点相邻左侧点的概率
            float xOffset = (ParamsAssessUtils::sign((rProb - lProb)) * 0.25);

            float uProb = output[c * height * width + width * (maxPoint.y + 1) + maxPoint.x];  // 该点上方相邻点的概率（左上角为原点）
            float dProb = output[c * height * width + width * (maxPoint.y - 1) + maxPoint.x];  // 该点下方相邻点的概率
            float yOffset = (ParamsAssessUtils::sign((uProb - dProb)) * 0.25);

            floatMaxPoint.x = xOffset + maxPoint.x;
            floatMaxPoint.y = yOffset + maxPoint.y;
            MaxPoints.push_back(floatMaxPoint);
            //maxPoint.x = static_cast<int>((maxPoint.x + xOffset) * 4.);  // 将修正完的坐标映射回resize之后的图片坐标
            //maxPoint.y = static_cast<int>((maxPoint.y + yOffset) * 4.);
        }
        else
        {
            MaxPoints.push_back(maxPoint);
        }
    }
    //int datatype = m_invertTrans.type();

    eaPoints = ImageProcess::affinePoints(MaxPoints, m_invertTrans);

    //cv::Mat dst;
    ////cv::Size inputSize(m_inputDims.d[3], m_inputDims.d[2]);

    //src.copyTo(dst);
    //cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);
    ////cv::resize(dst, dst, inputSize);

    //cv::line(dst, transedPoints[0], transedPoints[1], cv::Scalar(0, 255, 0), 2);
    //float avadDst = Utils::calcLineDist(transedPoints);
    //values.insert({ "Aortic Annulusd", avadDst });
    //resultPics.insert({ "Aortic Annulusd" , dst });

    //cv::imshow("test", dst);
    //cv::waitKey(0);

    return 1;
}
