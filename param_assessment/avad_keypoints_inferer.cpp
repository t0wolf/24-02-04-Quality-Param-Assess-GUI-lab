#include "avad_keypoints_inferer.h"

AVADKeypointsInferer::AVADKeypointsInferer(std::string& sEnginePath)  // 继承SegmnetInferBase的构造函数，调整输入输出的shape和classes
	:SegmentInferBase(sEnginePath)
    , m_scale(-10000.0f)
    , m_scaleLength(-1000.0f)
{
	m_inputDims = { 1, 3, 384, 512 };  // {height, width}
	m_outputDims = { 1, 2, 96, 128 };
	m_classes = 2;
}

AVADKeypointsInferer::~AVADKeypointsInferer()   // 释放两个指针
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

//int AVADKeypointsInferer::setScaleInfo(float& scaleLength, float& scale)
//{
//    m_scaleLength = scaleLength;
//    m_scale = scale;
//    return 1;
//}
int AVADKeypointsInferer::doInference(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics,float m_scale)
{
	int ret = doSingleInfer(src, values, resultPics, m_scale);
	return ret;
}

int AVADKeypointsInferer::doSingleInfer(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics, float m_scale)
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

    postProcess(output, src, values, resultPics,m_scale);  // 后处理，调用postProcess函数对输出数据进行后处理

    delete[] blob;
    delete[] output;
    cudaFree(inputMem);
    cudaFree(outputMem);
    cudaStreamDestroy(stream);
    return 1;
}

int AVADKeypointsInferer::preprocess(cv::Mat& src, float* blob)  // 图像预处理，并存入内存
{
    cv::Mat dst;
    std::pair<cv::Mat, cv::Mat> transPair;
    src.copyTo(dst);

    int fixedSize[2] = { m_inputDims.d[2], m_inputDims.d[3] };  // {h, w}
    //cv::Size inputSize(m_inputDims.d[3], m_inputDims.d[2]);   //size(width,height)
    cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);
    //cv::resize(dst, dst, inputSize);

    transPair = ImageProcess::affineTransform(dst, fixedSize);

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

int AVADKeypointsInferer::blobFromImage(cv::Mat& src, float* blob)  // 遍历图像，将每一个像素存到一维数组当中(每一个c，按照其上一行一行的顺序存进去)
{
    int channels = src.channels();
    int rows = src.rows;  // cols == width == Point.x;   rows == heigh == Point.y
    int cols = src.cols;
    float mean[3] = { 0.485f, 0.456f, 0.406f };
    float std[3] = { 0.229f, 0.224f, 0.225f };

    for (int c = 0; c < channels; c++)
    {
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                blob[c * rows * cols + row * cols + col] = (src.at<cv::Vec3f>(row, col)[c] / 255.0f - mean[c]) / std[c];  // to_tensor
            }
        }
    }
    return 1;
}

int AVADKeypointsInferer::postProcess(float* output, cv::Mat src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics, float m_scale)
{
    int height = m_outputDims.d[2];  // 96
    int width = m_outputDims.d[3];  // 128
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

    std::vector<cv::Point> transedPoints = ImageProcess::affinePoints(MaxPoints, m_invertTrans);

    cv::Mat dst;
    //cv::Size inputSize(m_inputDims.d[3], m_inputDims.d[2]);

    src.copyTo(dst);
    cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);
    //cv::resize(dst, dst, inputSize);

    cv::line(dst, transedPoints[0], transedPoints[1], cv::Scalar(0, 255, 0), 2);
    float avadDst = ParamsAssessUtils::calcLineDist(transedPoints);
    //if (m_scale > 0 && m_scaleLength > 0) {
    std::string length = std::to_string(static_cast<int>(m_scaleLength));
    if (m_scale > 0) {
        std::string distance = std::to_string(avadDst / m_scale);
        std::vector<float> fDistance = { avadDst / m_scale };
        values.insert({ "AoD", fDistance });
        resultPics.insert({ "AoD" , dst });
    }

   // }

    //parseFinalResults(tempValues, realValues, tempResultPics);

    //cv::imshow("test", dst);
    //cv::waitKey(0);

    return 1;
}
//
//int AVADKeypointsInferer::parseFinalResults(std::map<std::string, std::vector<float>>& values, std::map<std::string, std::vector<float>>& realValues, std::map<std::string, cv::Mat>& resultPics)
//{
//    // 检查前三个键是否在范围内
//    std::vector<std::string> edKeys = { "AoD" };
//    //std::vector<std::string> esKeys = { "LAD" };
//    bool allInRange = true;
//
//    for (const auto& key : edKeys)
//    {
//        if (realValues.find(key) != realValues.end() && !realValues[key].empty())
//        {
//            float value = realValues[key][0]; // 取第一个值进行判断
//            if (value < m_referRange[key].first || value > m_referRange[key].second)
//            {
//                allInRange = false;
//                break;
//            }
//        }
//        else
//        {
//            allInRange = false;
//            break;
//        }
//    }
//
//    // 如果前三个值都在范围内
//    if (allInRange)
//    {
//        for (const auto& key : edKeys)
//        {
//            m_values[key] = values[key];
//        }
//        if (!resultPics.empty())  // 将ivs对应的图像对加入成员变量中
//        {
//            auto it = resultPics.begin();
//            m_resultPics[it->first] = it->second;
//        }
//    }
//
//    //for (const auto& key : esKeys)
//    //{
//    //    if (realValues.find(key) != realValues.end() && !realValues[key].empty())
//    //    {
//    //        float ladValue = realValues[key][0];
//    //        if (ladValue >= m_referRange[key].first && ladValue <= m_referRange[key].second)
//    //        {
//    //            m_values[key] = values[key];
//    //            if (resultPics.find(key) != resultPics.end())
//    //            {
//    //                m_resultPics[key] = resultPics[key];
//    //            }
//    //        }
//    //    }
//    //}
//
//    return 1;
//}