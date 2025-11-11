#ifndef VIEW_CLS_INFERER_H
#define VIEW_CLS_INFERER_H

#include <iostream>
#include <vector>
#include <fstream>
#include <NvInfer.h>
#include <cassert>
#include <numeric>

#include <opencv2/opencv.hpp>
#include "logger.h"

class ViewClsInferer
{
public:
    ViewClsInferer(std::string backbonePath, std::string swinHeadPath);

    int doInference(cv::Mat& src);

    int doInference(cv::Mat& src, float& fQualityScore);

    int doInference(cv::Mat& src, float& fQualityScore, bool& bIsSwitch);

    int clearFeatMemory();

    int getClipSize();

private:
    std::streampos parseEngineFile(std::string& enginePath, std::ifstream& engineFile, std::vector<char>& engineData);

    int preprocess(cv::Mat& src, float* blob);

    int blobFromImage(cv::Mat& src, float* blob);

    int blobFromFeatures(std::vector<float*> vBlobs, float* blob);

    int backboneSingleInfer(cv::Mat& src, float* feat);

    int swinHeadSingleInfer(float* probs);

    int swinHeadSingleInfer(float* probs, float* qualityScore);

    int swinHeadSingleInfer(float* probs, float* qualityScore, float* isSwitch);

    int postProcess(float* probs);

    int postProcess(float* probs, int length);

    inline std::pair<float, int> argmax(float* arr, int length = 10) {
        float maxVal = arr[0];
        int maxIndex = 0;

        for(int i = 1; i < length; i++) {
            if(arr[i] > maxVal) {
                maxVal = arr[i];
                maxIndex = i;
            }
        }

        return std::make_pair(maxVal, maxIndex);
    }

    inline size_t getMemorySize(const nvinfer1::Dims& dims, const int32_t elem_size)
    {
        return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()) * elem_size;
    }

    inline size_t getElementNum(const nvinfer1::Dims& dims)
    {
        // int temp = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
        return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
    }

    inline int clearAndReleaseFeats()
    {
        for (float* feat : m_vFeats)
        {
            delete[] feat;
            feat = nullptr;
        }
        m_vFeats.clear();

        return 1;
    }

protected:
    nvinfer1::Dims4 m_backboneInputDims = { 1, 3, 320, 320 };
    nvinfer1::Dims4 m_backboneOutputDims = /*{1, 2048, 8, 8}*/{ 1, 512, 10, 10 };
    nvinfer1::Dims4 m_swinHeadInputDims = /*{16, 2048, 8, 8}*/{ 24, 512, 10, 10 };
    nvinfer1::Dims2 m_swinHeadOutputDims = {1, 10};

    int m_swinInputSingleSize = m_swinHeadInputDims.d[1] * m_swinHeadInputDims.d[2] * m_swinHeadInputDims.d[3];
    // nvinfer1::ICudaEngine* m_engine;
    nvinfer1::IExecutionContext* m_backboneContext;
    nvinfer1::IExecutionContext* m_swinHeadContext;

    std::vector<float> m_means = {0.485f, 0.456f, 0.406f};
    std::vector<float> m_stds = {0.229f, 0.224f, 0.225f};

    int m_backboneInputIdx;
    int m_backboneOutputIdx;
    int m_swinHeadInputIdx;
    int m_swinHeadOutputIdx;

    int m_classes = 10;
    int m_clipSize = 24;
    std::vector<float*> m_vFeats;
};

#endif // VIEW_CLS_INFERER_H
