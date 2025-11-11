#include "OCRDet.h"

OCRDet::OCRDet(std::string sDetModelPath)
    : m_inputDims({ 1, 3, 1024, 1024 })
    , m_outputDims({ 1, 1, 1024, 1024 })
{
    // De-serialize engine from file
    
    std::ifstream engineFile(sDetModelPath, std::ios::binary);
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

    qInfo() << "OCRDet m_engine...";

    m_context = m_engine->createExecutionContext();
    assert(m_context != nullptr);

    qInfo() << "OCRDet m_context...";

    m_inputImageIdx = m_engine->getBindingIndex("x");
    m_outputMaskIdx0 = m_engine->getBindingIndex("sigmoid_0.tmp_0");
    m_outputMaskIdx1 = m_engine->getBindingIndex("sigmoid_0.tmp_0");
    //m_outputMaskIdx0 = m_engine->getBindingIndex("sigmoid_11.tmp_0");
    //m_outputMaskIdx1 = m_engine->getBindingIndex("tmp_36");
    qInfo() << "OCRDet";
    //m_context->setBindingDimensions(m_inputImageIdx, m_inputDims);
}

OCRDet::~OCRDet()
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

int OCRDet::doInference(cv::Mat& src, std::vector<cv::Rect>& vChBoxes)
{
    doSingleInfer(src, vChBoxes);
    return 1;
}

int OCRDet::preprocess(cv::Mat& src, float* blob, float& ratio_h, float& ratio_w)
{
    m_originImgSize = src.size();

    cv::Mat dst = src.clone();
    //dst = cv::Scalar::all(255) - dst;

    //cv::imshow("inverse", dst);
    //cv::waitKey(0);
    //cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);
    dbResizeImage(src, dst, 640, ratio_h, ratio_w);

    //resizeImage(src, dst);
    //cv::imshow("resized", dst);
    //cv::waitKey(0);
    //cv::destroyWindow("resized");
    //cv::resize(dst, dst, cv::Size(m_inputDims.d[2], m_inputDims.d[3]));
    dst.convertTo(dst, CV_32FC3);
    blobFromImage(dst, blob);

    return 1;
}

int OCRDet::preprocess(cv::Mat& src, cv::Mat& dst, float& ratio_h, float& ratio_w)
{
    m_originImgSize = src.size();

    dst = src.clone();
    //dst = cv::Scalar::all(255) - dst;

    //cv::imshow("inverse", dst);
    //cv::waitKey(0);
    //cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);
    dbResizeImage(src, dst, 1280, ratio_h, ratio_w);

    //resizeImage(src, dst);
    //cv::imshow("resized", dst);
    //cv::waitKey(0);
    //cv::destroyWindow("resized");
    //cv::resize(dst, dst, cv::Size(m_inputDims.d[2], m_inputDims.d[3]));
    dst.convertTo(dst, CV_32FC3);
    return 1;
}

int OCRDet::resizeImage(cv::Mat& src, cv::Mat& dst)
{
    cv::Size imgOriginSize = m_originImgSize;
    cv::Size targetSize = cv::Size(m_inputDims.d[2], m_inputDims.d[3]);

    // 计算纵横比
    double aspectRatioTarget = static_cast<double>(targetSize.width) / targetSize.height;
    double aspectRatioInput = static_cast<double>(src.cols) / src.rows;

    // 根据纵横比决定是宽度填充还是高度填充
    if (aspectRatioInput > aspectRatioTarget) {
        // 输入图像更宽，将根据高度进行缩放
        m_scaleX = m_scaleY = static_cast<double>(targetSize.width) / src.cols;
        m_offsetX = 0;
        m_offsetY = (targetSize.height - static_cast<int>(src.rows * m_scaleY)) / 2;
    }
    else {
        // 输入图像更高，将根据宽度进行缩放
        m_scaleX = m_scaleY = static_cast<double>(targetSize.height) / src.rows;
        m_offsetX = (targetSize.width - static_cast<int>(src.cols * m_scaleX)) / 2;
        m_offsetY = 0;
    }

    // 调整图像大小
    cv::Mat resizedImage;
    cv::resize(src, resizedImage, cv::Size(static_cast<int>(src.cols * m_scaleX), static_cast<int>(src.rows * m_scaleY)), 0, 0, cv::INTER_LINEAR);

    // 创建目标尺寸的图像并填充边缘
    dst = cv::Mat::zeros(targetSize.height, targetSize.width, src.type());
    cv::copyMakeBorder(resizedImage, dst, m_offsetY, targetSize.height - resizedImage.rows - m_offsetY, m_offsetX, targetSize.width - resizedImage.cols - m_offsetX, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    return 1;
}

int OCRDet::dbResizeImage(cv::Mat& src, cv::Mat& dst, int max_size_len, float& ratio_h, float& ratio_w)
{
    int w = src.cols;
    int h = src.rows;

    float ratio = 1.f;
    int max_wh = w >= h ? w : h;
    //处理图像最长处不能超出预设值
    if (max_wh > max_size_len) {
        if (h > w) {
            ratio = float(max_size_len) / float(h);
        }
        else {
            ratio = float(max_size_len) / float(w);
        }
    }
    else {
        if (h > w) {
            ratio = float(max_size_len) / float(h);
        }
        else {
            ratio = float(max_size_len) / float(w);
        }
    }

    int resize_h = int(float(h) * ratio);
    int resize_w = int(float(w) * ratio);

    //除32，余下的超过16补全为32，不足16的长度舍弃
    resize_h = std::max(int(round(float(resize_h) / 32) * 32), 32);
    resize_w = std::max(int(round(float(resize_w) / 32) * 32), 32);

    cv::resize(src, dst, cv::Size(resize_w, resize_h));
    ratio_h = float(resize_h) / float(h);
    ratio_w = float(resize_w) / float(w);
    return 1;
}

int OCRDet::boxesProjBack(cv::Rect& detectedBox)
{
    int originalX = static_cast<int>((detectedBox.x - m_offsetX) / m_scaleX);
    int originalY = static_cast<int>((detectedBox.y - m_offsetY) / m_scaleY);
    int originalWidth = static_cast<int>(detectedBox.width / m_scaleX);
    int originalHeight = static_cast<int>(detectedBox.height / m_scaleY);

    detectedBox = cv::Rect(originalX, originalY, originalWidth, originalHeight);
    return 1;
}

int OCRDet::boxesProjBack(cv::Rect& detectedBox, float& ratio_h, float& ratio_w)
{
    int originalX = static_cast<int>(detectedBox.x / ratio_w);
    int originalY = static_cast<int>(detectedBox.y / ratio_h);
    int originalWidth = static_cast<int>(detectedBox.width / ratio_w);
    int originalHeight = static_cast<int>(detectedBox.height / ratio_h);

    detectedBox = cv::Rect(originalX, originalY, originalWidth, originalHeight);
    return 1;
}

int OCRDet::expandROI(cv::Rect& rect, int expandSize)
{
    int newX = (std::max)(rect.x - expandSize, 0);
    int newY = (std::max)(rect.y - expandSize, 0);
    int newWidth = (std::min)(rect.width + 2 * expandSize, m_originImgSize.width - newX);
    int newHeight = (std::min)(rect.height + 2 * expandSize, m_originImgSize.height - newY);

    rect = cv::Rect(newX, newY, newWidth, newHeight);
    return 1;
}

int OCRDet::blobFromImage(cv::Mat& src, float* blob)
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
                blob[c * rows * cols + row * cols + col] = (src.at<cv::Vec3f>(row, col)[c] / 255.0f - m_means[c]) / m_stds[c];
            }
        }
    }
    return 1;
}

int OCRDet::doSingleInfer(cv::Mat& src, std::vector<cv::Rect>& vChBoxes)
{
    float ratio_h, ratio_w;
    cv::Mat resized;
    preprocess(src, resized, ratio_h, ratio_w);

    void* inputMem{ nullptr };
    void* outputMem0{ nullptr };
    void* outputMem1{ nullptr };

    nvinfer1::Dims4 inputDims{1, 3, resized.rows, resized.cols};
    nvinfer1::Dims4 outputDims{ 1, 3, resized.rows, resized.cols };
    m_context->setBindingDimensions(m_inputImageIdx, inputDims);
    size_t inputSize = getMemorySize(inputDims, sizeof(float));
    size_t outputSize = getMemorySize(outputDims, sizeof(float));
    //size_t inputSize = std::accumulate(inputDims.d, inputDims.d + inputDims.nbDims, 1, std::multiplies<int64_t>()) * sizeof(float);
    //size_t outputSize = std::accumulate(inputDims.d, inputDims.d + inputDims.nbDims, 1, std::multiplies<int64_t>()) * sizeof(float);

    if (cudaMalloc(&inputMem, inputSize) != cudaSuccess)  // 使用cudaMalloc函数在GPU上为输入数据分配内存，将分配的内存地址存储在inputMem中。
    {
        logger::gLogError << "ERROR: input cuda memory allocation failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }
    if (cudaMalloc(&outputMem0, outputSize) != cudaSuccess)  // 使用cudaMalloc函数在GPU上为输出数据分配内存，将分配的内存地址存储在outputMem中。
    {
        logger::gLogError << "ERROR: output cuda memory allocation failed, size = " << outputSize << " bytes" << std::endl;
        return 0;
    }
    if (cudaMalloc(&outputMem1, outputSize) != cudaSuccess)  // 使用cudaMalloc函数在GPU上为输出数据分配内存，将分配的内存地址存储在outputMem中。
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

    float* blob = new float[getElementNum(inputDims)];
    float* output0 = new float[getElementNum(outputDims)];
    float* output1 = new float[getElementNum(outputDims)];

    blobFromImage(resized, blob);

    if (cudaMemcpyAsync(inputMem, blob, inputSize, cudaMemcpyHostToDevice, stream) != cudaSuccess)  // 使用cudaMemcpyAsync函数将blob中的数据异步拷贝到GPU的输入内存inputMem中。
    {
        logger::gLogError << "ERROR: CUDA memory copy of input failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }

    // Run TensorRT inference
    void* bindings[] = { inputMem, outputMem0, outputMem1 };
    bool status = m_context->enqueueV2(bindings, stream, nullptr);  // 执行推理
    if (!status)
    {
        logger::gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return 0;
    }

    if (cudaMemcpyAsync(output0, outputMem0, outputSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << outputSize << " bytes" << std::endl;
        return 0;
    }
    if (cudaMemcpyAsync(output1, outputMem1, outputSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << outputSize << " bytes" << std::endl;
        return 0;
    }

    postProcess(output0, outputDims, vChBoxes, ratio_h, ratio_w);

    delete[] blob;
    delete[] output0;
    delete[] output1;

    cudaFree(inputMem);
    cudaFree(outputMem0);
    cudaFree(outputMem1);
    cudaStreamDestroy(stream);
    return 1;
}

int OCRDet::postProcess(float* output, nvinfer1::Dims4& outputDims, std::vector<cv::Rect>& vChBoxes, float& ratio_h, float& ratio_w)
{
    cv::Mat mask;
    argmaxMask(output, outputDims, mask);
    getBoxesFromMask(mask, vChBoxes, ratio_h, ratio_w);

    return 1;
}

int OCRDet::argmaxMask(float* output, nvinfer1::Dims4& outputDims, cv::Mat& mask)
{
    cv::Size outputSize(outputDims.d[3], outputDims.d[2]);
    int height = outputSize.height;
    int width = outputSize.width;

    if (mask.empty())
        mask = cv::Mat::zeros(outputSize, CV_8UC1);

    // Iterate through the mask data and assign pixel values to the resultMat

    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            float currProb = output[width * h + w];
            if (currProb > 0.1f)
            {
                mask.at<uchar>(h, w) = 255;
            }
        }
    }

    //cv::imshow("mask", mask);
    //cv::waitKey(0);
    //cv::destroyWindow("mask");
    return 1;
}

int OCRDet::getBoxesFromMask(cv::Mat& mask, std::vector<cv::Rect>& vChBoxes, float& ratio_h, float& ratio_w)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    int kernelSize = 3;
    if (ratio_h >= 2.0f)
        kernelSize = 11;

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    //cv::imshow("mask", mask);
    //cv::waitKey(0);

    cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    contours = filterAbnormalSmallContours(contours);
    int nContours = contours.size();

    int img_w = mask.cols;
    int img_h = mask.rows;

    for (auto& contour : contours)
    {
        std::vector<cv::Point> boxes;
        getMinimizeBoxes(contour, boxes);

        for (auto& point : boxes)
        {
   //         int x = static_cast<int>(point.x / scaleW);
			//int y = static_cast<int>(point.y / scaleH);
            int x = static_cast<int>(point.x);
            int y = static_cast<int>(point.y);

            x = static_cast<int>(x / ratio_w);
            y = static_cast<int>(y / ratio_h);

            x = (std::max)((std::min)(x, (m_originImgSize.width - 1)), 0);
            y = (std::max)((std::min)(y, (m_originImgSize.height - 1)), 0);
            point.x = x;
            point.y = y;
        }
        cv::Rect chBoundRect = cv::boundingRect(boxes);
        //int originalX = static_cast<int>(chBoundRect.x / ratio_w);
        //int originalY = static_cast<int>(chBoundRect.y / ratio_h);
        //originalX = (std::max)((std::min)(originalX, (m_originImgSize.width - 1)), 0);
        //originalY = (std::max)((std::min)(originalY, (m_originImgSize.height - 1)), 0);

        //int originalWidth = static_cast<int>(chBoundRect.width / ratio_w);
        //int originalHeight = static_cast<int>(chBoundRect.height / ratio_h);

        //detectedBox = cv::Rect(originalX, originalY, originalWidth, originalHeight);

        //boxesProjBack(chBoundRect, ratio_h, ratio_w);
        expandROI(chBoundRect, 4);

        vChBoxes.push_back(chBoundRect);
    }

    return 1;
}

int OCRDet::getMinimizeBoxes(std::vector<cv::Point>& contours, std::vector<cv::Point>& miniBoxes)
{
    cv::RotatedRect minRect = cv::minAreaRect(contours);
    cv::Point2f rectPoints[4];
    minRect.points(rectPoints);

    std::vector<cv::Point2f> sortedPoints(rectPoints, rectPoints + 4);
    std::sort(sortedPoints.begin(), sortedPoints.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.x < b.x;
    });

    int idx1 = 0;
    int idx2 = 0;
    int idx3 = 0;
    int idx4 = 0;

    if (sortedPoints[1].y > sortedPoints[0].y)
    {
		idx1 = 0;
		idx4 = 1;
	}
    else
    {
		idx1 = 1;
		idx4 = 0;
	}

    if (sortedPoints[3].y > sortedPoints[2].y)
    {
        idx2 = 2;
        idx3 = 3;
    }
    else
    {
        idx2 = 3;
        idx3 = 2;
    }

    miniBoxes = { sortedPoints[idx1], sortedPoints[idx2], sortedPoints[idx3], sortedPoints[idx4] };

    return 1;
}

std::vector<std::vector<cv::Point>> OCRDet::filterAbnormalSmallContours(std::vector<std::vector<cv::Point>>& contours)
{
    std::vector<std::vector<cv::Point>> resultContours;

    for (auto& contour : contours)
    {
        float currArea = cv::contourArea(contour);
        if (currArea >= 3.0f)
            resultContours.push_back(contour);
    }
    return resultContours;
}
