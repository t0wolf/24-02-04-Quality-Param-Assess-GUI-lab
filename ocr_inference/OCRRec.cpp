#include "OCRRec.h"

OCRRec::OCRRec(std::string sDetModelPath, std::string dictionaryPath)
    //: m_inputDims({ 1, 3, 32, 320 })
    : m_inputDims({ 1, 3, 48, 320 })
    //, m_outputDims({ 1, 80, 6625 })
    , m_outputDims({ 1, 40, 97 })
{
    //qInfo() << "OCRRec...";
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

    m_context = m_engine->createExecutionContext();
    assert(m_context != nullptr);

    m_inputImageIdx = m_engine->getBindingIndex("x");
    
    m_outputMaskIdx = m_engine->getBindingIndex("softmax_2.tmp_0");  // en
    //m_outputMaskIdx = m_engine->getBindingIndex("softmax_1.tmp_0");  // ch

    m_context->setBindingDimensions(m_inputImageIdx, m_inputDims);
    
    parseDict(dictionaryPath);
    //qInfo() << "OCRRec";
}

OCRRec::~OCRRec()
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

int OCRRec::doInference(std::vector<cv::Mat>& vSrc, std::vector<std::string>& vChTexts)
{
    for (auto& src : vSrc)
    {
        //cv::imshow("src", src);
        //cv::waitKey(0);
		std::string vChText;
		doSingleInfer(src, vChText);
		vChTexts.push_back(vChText);
        qDebug() << QString::fromStdString(vChText);
        //if (vChText == "Ning, Weiping" || vChText == "250814189996")
        //{
        //    cv::imshow("src", src);
        //    cv::waitKey(0);
        //}
    }
    return 1;
}

int OCRRec::preprocess(cv::Mat& src, float* blob)
{
    m_originImgSize = src.size();
    cv::Mat dst = src.clone();

    //cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);
    int imgH = m_inputDims.d[2];
    int imgW = m_inputDims.d[3];
    float whRatio = imgW * 1.0f / imgH;

    crnnResizeImg(src, dst, whRatio);
    Normalize(&dst);
    //resizeImg(src, dst);
    //cv::resize(dst, dst, cv::Size(m_inputDims.d[3], m_inputDims.d[2]));

    dst.convertTo(dst, CV_32FC3);
    blobFromImage(dst, blob, false);

    return 1;
}

int OCRRec::resizeImg(cv::Mat& src, cv::Mat& dst)
{
    if (src.cols * 1.0 / src.rows * m_inputDims.d[2] >= m_inputDims.d[3])
    {
        cv::resize(src, dst, cv::Size(m_inputDims.d[3], m_inputDims.d[2]));
        //cv::imshow("img", dst);
        //cv::waitKey(0);
        Normalize(&dst);
    }
    else
    {
        cv::resize(src, dst, cv::Size(int(src.cols * 1.0 / src.rows * m_inputDims.d[2] + 1), m_inputDims.d[2]), 0.f, 0.f,
            cv::INTER_LINEAR);
        //cv::imshow("img", dst);
        //cv::waitKey(0);
        Normalize(&dst);
        cv::copyMakeBorder(dst, dst, 0, 0, 0, m_inputDims.d[3] - int(src.cols * 1.0 / src.rows * m_inputDims.d[2] + 1), cv::BORDER_CONSTANT, { 0, 0, 0 });
    }
    return 1;
}

void OCRRec::crnnResizeImg(const cv::Mat& img, cv::Mat& resize_img, float wh_ratio)
{
    int imgC, imgH, imgW;
    imgC = m_inputDims.d[1];
    imgH = m_inputDims.d[2];
    imgW = m_inputDims.d[3];

    imgW = int(imgH * wh_ratio);

    float ratio = float(img.cols) / float(img.rows);
    int resize_w, resize_h;


    if (ceilf(imgH * ratio) > imgW)
        resize_w = imgW;
    else
        resize_w = int(ceilf(imgH * ratio));

    cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
        cv::INTER_LINEAR);
    // 由于图像宽高比小于预设值，因此在高度达到预设值后，需要在右侧进行填充
    cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0,
        int(imgW - resize_img.cols), cv::BORDER_CONSTANT,
        { 127, 127, 127 });
}

int OCRRec::postProcess(float* output, std::string& vChTexts)
{
    int lastIdx = 0;
    int counter = 0;
    float fScore = 0.0f;
    vChTexts.clear();
    int nClasses = m_outputDims.d[2];
    int nTokens = m_outputDims.d[1];

    for (int n = 0; n < nTokens; n++)
    {
        int maxIdx = 0;
        float maxVal = -100.0f;
        for (int c = 0; c < nClasses; c++)
        {
            if (output[n * nClasses + c] > maxVal)
            {
				maxVal = output[n * nClasses + c];
				maxIdx = c;
			}
        }
        if (maxIdx > 0 && (!(n > 0 && maxIdx == lastIdx)))
        {
            fScore += maxVal;
            counter += 1;
            if (maxIdx - 1 >= m_dictionary.size())
                vChTexts += " ";
            else
                vChTexts = vChTexts + m_dictionary[maxIdx - 1];
        }
        lastIdx = maxIdx;
    }
    fScore /= counter;
    return 1;
}

int OCRRec::blobFromImage(cv::Mat& src, float* blob)
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
                //blob[c * rows * cols + row * cols + col] = (src.at<cv::Vec3f>(row, col)[c] / 255.0f - m_means[c]) / m_stds[c];
                //float tempVal = (src.at<cv::Vec3f>(row, col)[c] / 255.0f - 0.5f) / 0.5f;
                blob[c * rows * cols + row * cols + col] = (src.at<cv::Vec3f>(row, col)[c] / 255.0f - 0.5f) / 0.5f;
            }
        }
    }
    return 1;
}

int OCRRec::blobFromImage(cv::Mat& src, float* blob, bool normalize)
{
    if (normalize)
    {
        blobFromImage(src, blob);
        return 1;
    }
    else
    {
        int channels = src.channels();
        int rows = src.rows;
        int cols = src.cols;

        for (int c = 0; c < channels; c++)
        {
            for (int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++)
                {
                    blob[c * rows * cols + row * cols + col] = src.at<cv::Vec3f>(row, col)[c];
                }
            }
        }

        return 1;
    }
}

int OCRRec::doSingleInfer(cv::Mat& src, std::string& vChTexts)
{
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

    float* blob = new float[getElementNum(m_inputDims)];
    float* output = new float[getElementNum(m_outputDims)];
    preprocess(src, blob);

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

    if (cudaMemcpyAsync(output, outputMem, outputSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << outputSize << " bytes" << std::endl;
        return 0;
    }

    postProcess(output, vChTexts);

    delete[] blob;
    delete[] output;
    cudaFree(inputMem);
    cudaFree(outputMem);
    cudaStreamDestroy(stream);
    return 1;
}

int OCRRec::parseDict(std::string dictionaryPath)
{
    qInfo() << "Parsing Dict...";
    std::ifstream in(dictionaryPath);
    std::string line;
    if (in)
    {
        while (getline(in, line))
        {
            m_dictionary.push_back(line);
        }
    }
    else
    {
        return 0;
    }

    qInfo() << "Parsed.";
	return 1;
}

//int OCRRec::parseChineseDict(std::string& dictionaryPath)
//{
//    std::wifstream file(dictionaryPath);
//    if (!file.is_open()) {
//        return 0;
//    }
//
//    // 设置locale以支持UTF-8（假设文件是UTF-8编码）
//    file.imbue(std::locale(file.getloc(), new std::codecvt_utf8<wchar_t>));
//
//    std::wstring line;
//    while (std::getline(file, line)) {
//        // 假设字典每一行都是一个汉字或标记
//        if (!line.empty()) {
//            m_dictionary.push_back(line);
//        }
//    }
//
//    file.close();
//
//    return 1;
//}
