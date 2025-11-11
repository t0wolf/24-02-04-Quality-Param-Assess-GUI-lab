#include "roi_detection.h"

ROIDetection::ROIDetection(std::string sEngineFile, cv::Size imgSize)
    : m_inputDims(nvinfer1::Dims4({1, 3, imgSize.height, imgSize.width}))
    , m_classes(2)
    , m_imgSize(imgSize)
    , m_fConfThresh(0.1f)
{
    m_outputDims = nvinfer1::Dims3({1, 25200, (m_classes + 5)});
    // De-serialize engine from file
    std::ifstream engineFile(sEngineFile, std::ios::binary);
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

    m_inputImageIdx = m_engine->getBindingIndex("images");
    m_output1Idx = m_engine->getBindingIndex("onnx::Sigmoid_353");
    m_output2Idx = m_engine->getBindingIndex("onnx::Sigmoid_419");
    m_output3Idx = m_engine->getBindingIndex("onnx::Sigmoid_485");
    m_outputMaskIdx = m_engine->getBindingIndex("output");

    //m_context->setBindingDimensions(m_inputImageIdx, m_inputDims);
}

ROIDetection::~ROIDetection()
{
    delete m_engine;
    delete m_context;
}

int ROIDetection::preprocess(cv::Mat& src, cv::Mat& dst, float* blob)
{
    m_imgOriginSize = src.size();
    cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
    cv::resize(dst, dst, m_imgSize);
    dst.convertTo(dst, CV_32FC3);

    dst = dst / 255.0f;
    blobFromImage(dst, blob);

    return 1;
}

int ROIDetection::blobFromImage(cv::Mat& src, float* blob)
{
    int channels = src.channels();
    int cols = src.cols;
    int rows = src.rows;

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

std::vector<Object> ROIDetection::doInference(cv::Mat& src)
{
    float* blob = new float[getElementNum(m_inputDims)];
    float* output = new float[getElementNum(m_outputDims)];

    doSingleInfer(src, blob, output);

    std::vector<Object> objects, results;

    postProcess(output, objects, results);

    delete[] blob;
    delete[] output;

    cv::Mat drawFrame = src.clone();

    //for (auto & result : results)
    //    cv::rectangle(drawFrame, result.rect, cv::Scalar(0, 255, 0), 2);
    //cv::rectangle(drawFrame, roiScaleInfo.specScaleRect, cv::Scalar(0, 255, 0), 2);
    //cv::imshow("spec scale", drawFrame);
    //cv::waitKey(1);

    return results;
}

int ROIDetection::doSingleInfer(cv::Mat& src, float* blob, float* output)
{
    void* inputMem{ nullptr };
    void* outputMem{ nullptr };

    // intermediate output ptr
    void* outputMem1{ nullptr };
    void* outputMem2{ nullptr };
    void* outputMem3{ nullptr };

    size_t inputSize = getMemorySize(m_inputDims, sizeof(float));
    size_t outputSize = getMemorySize(m_outputDims, sizeof(float));

    if (cudaMalloc(&inputMem, inputSize) != cudaSuccess)
    {
        //logger::gLogError << "ERROR: input cuda memory allocation failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }

    nvinfer1::Dims4 outputDims1{ 3, 80, 80, 7 };
    size_t outputSize1 = getMemorySize(outputDims1, sizeof(float));
    if (cudaMalloc(&outputMem1, outputSize1) != cudaSuccess)
    {
        //logger::gLogError << "ERROR: output cuda memory allocation failed, size = " << outputSize << " bytes" << std::endl;
        return 0;
    }

    nvinfer1::Dims4 outputDims2{ 3, 40, 40, 7 };
    size_t outputSize2 = getMemorySize(outputDims2, sizeof(float));
    if (cudaMalloc(&outputMem2, outputSize2) != cudaSuccess)
    {
        //logger::gLogError << "ERROR: output cuda memory allocation failed, size = " << outputSize << " bytes" << std::endl;
        return 0;
    }

    nvinfer1::Dims4 outputDims3{ 3, 20, 20, 7 };
    size_t outputSize3 = getMemorySize(outputDims3, sizeof(float));
    if (cudaMalloc(&outputMem3, outputSize) != cudaSuccess)
    {
        //logger::gLogError << "ERROR: output cuda memory allocation failed, size = " << outputSize << " bytes" << std::endl;
        return 0;
    }

    if (cudaMalloc(&outputMem, outputSize) != cudaSuccess)
    {
        //logger::gLogError << "ERROR: output cuda memory allocation failed, size = " << outputSize << " bytes" << std::endl;
        return 0;
    }

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        //logger::gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return 0;
    }

    cv::Mat dst = src.clone();
    preprocess(src, dst, blob);

    if (cudaMemcpyAsync(inputMem, blob, inputSize, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        //logger::gLogError << "ERROR: CUDA memory copy of input failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }

    // Run TensorRT inference
    //void* bindings[] = { inputMem, outputMem1, outputMem2, outputMem3, outputMem };
    void* bindings[] = { inputMem, outputMem };
    bool status = m_context->enqueueV2(bindings, stream, nullptr);
    if (!status)
    {
        //logger::gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return 0;
    }

    if (cudaMemcpyAsync(output, outputMem, outputSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << outputSize << " bytes" << std::endl;
        return 0;
    }

    // postProcess(output, mask);

    cudaFree(inputMem);
    cudaFree(outputMem);
    cudaFree(outputMem1);
    cudaFree(outputMem2);
    cudaFree(outputMem3);
    cudaStreamDestroy(stream);

    return 1;
}

int ROIDetection::postProcess(float* output, std::vector<Object>& objects, std::vector<Object>& results)
{
    generateProposals(output, objects);
    qsort_descent_inplace(objects);

    std::vector<int> picked;
    nms_sorted_bboxes(objects, picked);

    int count = picked.size();

    int img_w = m_imgOriginSize.width;
    int img_h = m_imgOriginSize.height;
    float scaleH =  static_cast<float>(m_imgSize.height) / static_cast<float>(img_h);
    float scaleW =  static_cast<float>(m_imgSize.width) / static_cast<float>(img_w);

    if (!results.empty())
        results.clear();

    results.resize(count);
    for (int i = 0; i < count; i++)
    {
        Object obj = objects[picked[i]];

        // adjust offset to original unpadded
        float x0 = static_cast<float>(obj.rect.x) / scaleW;
        float y0 = static_cast<float>(obj.rect.y) / scaleH;
        float x1 = static_cast<float>(obj.rect.x + obj.rect.width) / scaleW;
        float y1 = static_cast<float>(obj.rect.y + obj.rect.height) / scaleH;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.0f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.0f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.0f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.0f);

        obj.rect.x = static_cast<int>(x0);
        obj.rect.y = static_cast<int>(y0);
        obj.rect.width = static_cast<int>(x1 - x0);
        obj.rect.height = static_cast<int>(y1 - y0);
        results[i] = obj;
    }
    return 1;
}

int ROIDetection::generateProposals(float* output, std::vector<Object>& objects)
{
    for (int i = 0; i < 25200; i++)
    {
        float conf = output[i * (m_classes + 5) + 4];
        if (conf > m_fConfThresh)
        {
            Object obj;
            float cx = output[i * (m_classes + 5)];
            float cy = output[i * (m_classes + 5) + 1];
            float w  = output[i * (m_classes + 5) + 2];
            float h  = output[i * (m_classes + 5) + 3];
            obj.rect.x = static_cast<int>(cx - w * 0.5f);
            obj.rect.y = static_cast<int>(cy - h * 0.5f);
            obj.rect.width = static_cast<int>(w);
            obj.rect.height = static_cast<int>(h);

            std::vector<float> vSingleProbs(m_classes);
            for (int j = 0; j < vSingleProbs.size(); j++)
            {
                vSingleProbs[j] = output[i * (m_classes + 5) + 5 + j];
            }

            auto max = argmax(vSingleProbs);
            obj.label = max.first;
            obj.conf = conf;

            objects.push_back(obj);
        }
    }
    return 1;
}

void ROIDetection::qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].conf;

    while (i <= j)
    {
        while (objects[i].conf > p)
            i++;

        while (objects[j].conf < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

void ROIDetection::qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

void ROIDetection::nms_sorted_bboxes(const std::vector<Object>& vObjects, std::vector<int>& picked)
{
    picked.clear();

    const int n = vObjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = vObjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = vObjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = vObjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > m_fNMSThresh)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

void ROIDetection::prepareIntermediateOutput()
{

}
