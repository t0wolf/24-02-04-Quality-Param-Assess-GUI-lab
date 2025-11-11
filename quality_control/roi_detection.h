#ifndef ROIDETECTION_H
#define ROIDETECTION_H
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <numeric>
#include <cassert>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include "logger.h"
#include "../type_define.h"

//struct Object
//{
//    cv::Rect rect;
//    int label;
//    float conf;
//};


class ROIDetection
{
public:
    ROIDetection(std::string sEngineFile, cv::Size imgSize);

    ~ROIDetection();

    std::vector<Object> doInference(cv::Mat& src);

private:
    int preprocess(cv::Mat& src, cv::Mat& dst, float* blob);

    int blobFromImage(cv::Mat& src, float* blob);

    int doSingleInfer(cv::Mat& src, float* blob, float* output);

    int postProcess(float* output, std::vector<Object>& objects, std::vector<Object>& results);

    int generateProposals(float* output, std::vector<Object>& objects);

    void qsort_descent_inplace(std::vector<Object>& objects, int left, int right);

    void qsort_descent_inplace(std::vector<Object>& objects);

    void nms_sorted_bboxes(const std::vector<Object>& vObjects, std::vector<int>& picked);

    void prepareIntermediateOutput();

    static inline float intersection_area(const Object& a, const Object& b)
    {
        if (a.rect.x > b.rect.x + b.rect.width || a.rect.x + a.rect.width < b.rect.x || a.rect.y > b.rect.y + b.rect.height || a.rect.y + a.rect.height < b.rect.y)
        {
            // no intersection
            return 0.f;
        }

        float inter_width = (std::min)(a.rect.x + a.rect.width, b.rect.x + b.rect.width) - (std::max)(a.rect.x, b.rect.x);
        float inter_height = (std::min)(a.rect.y + a.rect.height, b.rect.y + b.rect.height) - (std::max)(a.rect.y, b.rect.y);

        return inter_width * inter_height;
    }

protected:
    nvinfer1::Dims4 m_inputDims;
    nvinfer1::Dims3 m_outputDims;
    nvinfer1::ICudaEngine* m_engine;
    nvinfer1::IExecutionContext* m_context;

    std::vector<float> m_means;
    std::vector<float> m_stds;

    int m_inputImageIdx;
    int m_outputMaskIdx;

    // intermediate output
    int m_output1Idx;
    int m_output2Idx;
    int m_output3Idx;
    
    cv::Size m_imgSize;
    cv::Size m_imgOriginSize;
    float m_fConfThresh;
    float m_fNMSThresh = 0.45f;

    int m_classes;

    std::pair<int, float> argmax(std::vector<float>& vSingleProbs)
    {
        std::pair<int, float> result;
        auto iter = std::max_element(vSingleProbs.begin(), vSingleProbs.end());
        result.first = static_cast<int>(iter - vSingleProbs.begin());
        result.second = *iter;

        return result;
    }

    inline size_t getMemorySize(const nvinfer1::Dims& dims, const int32_t elem_size)
    {
        return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()) * elem_size;
    }

    inline size_t getElementNum(const nvinfer1::Dims& dims)
    {
        return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
    }
};

#endif // ROIDETECTION_H
