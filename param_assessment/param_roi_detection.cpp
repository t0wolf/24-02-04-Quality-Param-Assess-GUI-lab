#include "param_roi_detection.h"

ParamROIDetection::ParamROIDetection(std::string sEngineFile)
    : m_inputDims(nvinfer1::Dims4({1, 3, 640, 640}))
    , m_classes(1)
    , m_imgSize(m_inputDims.d[3], m_inputDims.d[2])
    , m_fConfThresh(0.4f)
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
    engineFile.close();

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger::gLogger.getTRTLogger());
    m_engine = runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
    assert(m_engine != nullptr);

    m_context = m_engine->createExecutionContext();
    assert(m_context != nullptr);

    m_inputImageIdx = m_engine->getBindingIndex("images");
    m_outputMaskIdx = m_engine->getBindingIndex("output0");

    m_context->setBindingDimensions(m_inputImageIdx, m_inputDims);
}

ParamROIDetection::~ParamROIDetection()
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

int ParamROIDetection::preprocess(cv::Mat& src, float* blob)
{
    m_imgOriginSize = src.size();
    cv::Mat dst = src.clone();
    cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
    cv::resize(dst, dst, m_imgSize);
    dst.convertTo(dst, CV_32FC3);

    dst = dst / 255.0f;
    blobFromImage(dst, blob);

    return 1;
}

int ParamROIDetection::blobFromImage(cv::Mat& src, float* blob)
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

std::vector<Object> ParamROIDetection::doInference(cv::Mat& src)
{
    float* blob = new float[getElementNum(m_inputDims)];
    float* output = new float[getElementNum(m_outputDims)];
    //cv::imshow("test", src);
    //cv::waitKey(0);

    doSingleInfer(src, blob, output);

    std::vector<Object> objects, results;

    postProcess(output, objects, results);

    delete[] blob;
    delete[] output;

    return results;
}

int ParamROIDetection::doSingleInfer(cv::Mat& src, float* blob, float* output)
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

    // postProcess(output, mask);

    cudaFree(inputMem);
    cudaFree(outputMem);
    cudaStreamDestroy(stream);

    return 1;
}

int ParamROIDetection::postProcess(float* output, std::vector<Object>& objects, std::vector<Object>& results)
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

int ParamROIDetection::generateProposals(float* output, std::vector<Object>& objects)
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

void ParamROIDetection::qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
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

void ParamROIDetection::qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

void ParamROIDetection::nms_sorted_bboxes(const std::vector<Object>& vObjects, std::vector<int>& picked)
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
