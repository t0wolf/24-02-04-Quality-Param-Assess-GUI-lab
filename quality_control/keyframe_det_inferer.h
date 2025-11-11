#ifndef KEYFRAMEDETINFERER_H
#define KEYFRAMEDETINFERER_H
#define NOMINMAX

#include "cuda_runtime_api.h"
#include <algorithm>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "logger.h"
#include "NvCaffeParser.h"

#include "cplusfuncs.h"
#include "typemappings.h"
#include <algorithm>
#include "type_define.h"
#include <QDebug>
#include <thread>
#include <mutex>
#include "QtLogger.h"

using namespace nvinfer1;

class KeyframeDetInferer
{
public:
    KeyframeDetInferer(std::string sEnginePath, std::string sSGTAEnginePath, int staLength, int memoryLength, bool isEDOnly = true, bool isAvgProb = false,
        std::vector<float> vecMeans = { 114.7748f, 107.7354f, 99.475f }, std::vector<float> vecStds = { 1.0f, 1.0f, 1.0f });

    int doInference(cv::Mat& src, std::string mode, std::vector<float>& vProbs);

    int doInference(cv::Mat& src, std::string mode, std::vector<float>& vecEDProbs, std::vector<float>& vecESProbs);

    int doInference(cv::Mat& src, std::string mode, std::vector<PeakInfo>& vPeaks);

    int clearFeatMemory()
    {
        std::lock_guard<std::mutex> locker(m_featsMutex);
        releaseFeatsVector();
        m_demoImages.clear();
        return 1;
    }

protected:
    int initializeBackboneEngine(std::string sBackboneEnginePath);

    int initializeSGTAEngine(std::string sSGTAEnginePath);

    int preprocess(cv::Mat& src, float* blob);

    int blobFromImage(cv::Mat& src, float* blob);

    int blobFromFeats(float* blobFeat);

    int blobFromQueryFeats(std::vector<float*>& vSampledFeats, float* blobFeat);

    int doBackboneInference(float* blob, float* feat);

    int doSGTAInference(float* blobFeat, float* feat, float* prob);

    virtual int calcEachFrameAvgProb(std::vector<std::vector<float>>& vecTotalProbs, std::vector<float>& vecAvgProbs);

    int sgtaInference(cv::Mat& src, std::vector<float>& vecEDIdxes, std::vector<float>& vecESIdxes);

    float checkEDESClose(std::vector<PeakInfo>& vecEDPeaks, std::vector<PeakInfo>& vecESPeaks);

    float checkProbClose(std::vector<float>& vecProbs);

    static float inline calcAverageValue(std::vector<float>& vProb)
    {
        return std::accumulate(vProb.begin(), vProb.end(), 0.0f) / static_cast<float>(vProb.size());
    }

    inline void releaseFeatsVector()
    {
        if (!m_vFeats.empty())
            m_vFeats.clear();
        if (!m_memFeats.empty())
            m_memFeats.clear();
    }

    inline size_t getMemorySize(const nvinfer1::Dims& dims, const int32_t elem_size)
    {
        return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()) * elem_size;
    }

    inline size_t getElementNum(const nvinfer1::Dims& dims)
    {
        return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
    }

    virtual inline std::vector<float*> getQueryFeat(int index)
    {
        int sampleNum = m_staLength / 2;
        std::vector<float*> sampledFeats;

        if (m_vFeats.empty())
            return sampledFeats;

        for (int i = -sampleNum; i < sampleNum + 1; i++)
        {
            int currIdx = index + i;
            currIdx = (std::min)((std::max)(currIdx, 0), static_cast<int>(m_vFeats.size() - 1));
            sampledFeats.push_back(m_vFeats[currIdx].get());
        }

        return sampledFeats;
    }

    virtual int getSampledElements(int element_count)
    {
        if (m_vFeats.empty())
            return 0;
        // 计算等间距的步长
        //int step = (std::max)(static_cast<int>(m_vFeats.size() / element_count), 1);

        //// 使用循环获取指定个数的元素
        //for (int i = 0; i < element_count; i++) {
        //    int index = i * step;
        //    // 当前索引超出向量长度时，取最后一个元素
        //    if (index >= m_vFeats.size()) {
        //        index = m_vFeats.size() - 1;
        //    }
        //    m_memFeats.push_back(m_vFeats[index]);
        //}

        // 顺序采样15帧
        //int nMaxIdx = m_vFeats.size() - 1;
        //if (nMaxIdx < element_count)
        //    return 0;

        //for (int i = 0; i < element_count; ++i)
        //{
        //    m_memFeats.push_back(m_vFeats[i]);
        //}

        // 20250423新增
        int nFeatArraySize = getElementNum(m_featDims);
        for (int i = 0; i < element_count; ++i)
        {
            if (i > m_vFeats.size() - 1)
            {
                floatArrayPtr lastElem = m_vFeats.back();
                floatArrayPtr newElem(new float[nFeatArraySize]);

                for (int j = 0; j < nFeatArraySize; ++j)
                {
                    newElem.get()[j] = lastElem.get()[j];
                }
                m_memFeats.push_back(newElem);
            }
            else
                m_memFeats.push_back(m_vFeats[i]);
        }

        return 1;
    }

    inline bool isPeak(const std::vector<float>& input, int index, float threshold) 
    {
        if (index == 0)
            return (input[index] > input[index + 1] + threshold && input[index] > input[index + 2] + threshold);
        else if (index == input.size() - 1)
            return (input[index] > input[index - 1] + threshold && input[index] > input[index - 2] + threshold);
        return input[index] > input[index - 1] + threshold && input[index] > input[index + 1] + threshold;
    }

    int findLocalMaxima(const std::vector<float>& vecInputs,
        std::vector<int>& vecPeakIdxes,
        std::vector<int>& vecLeftEdges,
        std::vector<int>& vecRightEdges);

    std::vector<PeakInfo> filterPeaksByDistance(std::vector<PeakInfo>& vecPeakInfos, int distance);

    std::vector<PeakInfo> filterPeaksByValue(std::vector<PeakInfo>& vecPeakInfos, float fHeight, float fThreshold);

    std::vector<PeakInfo> findPeaks(
        const std::vector<float>& input,
        float height = std::numeric_limits<float>::infinity(),
        size_t distance = 1,
        float threshold = 0,
        size_t plateau_size = 1
    );

    std::vector<int> findPeaks(std::vector<double> x,
                               std::vector<double> plateauSize = {},
                               std::vector<double> height = {},
                               std::vector<double> threshold = {},
                               int distance = 0, std::vector<double> prominence = {},
                               int wlen = 2, std::vector<double> width = {},
                               double relHeight = 0.5);
protected:
    // nvinfer1::Dims4 m_outputDims;
    // nvinfer1::ICudaEngine* m_engine;
    std::mutex m_featsMutex;

    nvinfer1::IExecutionContext* m_backboneContext;
    int m_inputImageIdx;
    int m_outputFeatIdx;
    nvinfer1::Dims4 m_inputDims;
    nvinfer1::Dims4 m_featDims;

    nvinfer1::IExecutionContext* m_sgtaContext;
    int m_inputMemFeatIdx;
    int m_inputQueryFeatIdx;
    int m_outputProbIdx;
    nvinfer1::Dims4 m_inputMemDims;
    nvinfer1::Dims4 m_inputQueryDims;
    nvinfer1::Dims2 m_outputProbDims;

    int m_staLength;
    int m_memLength;
    bool m_bIsEDOnly;
    bool m_bIsAvgProb;

    float m_fEDESProbDiffThresh = 0.1f;

    std::vector<float> m_means;
    std::vector<float> m_stds;

    //std::vector<float*> m_vFeats;
    //std::vector<float*> m_memFeats;
    std::vector<floatArrayPtr> m_vFeats;
    std::vector<floatArrayPtr> m_memFeats;
    std::vector<cv::Mat> m_demoImages;

    int m_counter;

    int m_classes;
};

class KeyframeLGTAInferer : public KeyframeDetInferer
{
public:
    KeyframeLGTAInferer(std::string strbackbonePath, std::string strLGTAEnginePath, int shortLength, int memoryLength, bool isEDOnly = true, bool isAvgProb = false,
        std::vector<float> vecMeans = { 114.7748f, 107.7354f, 99.475f }, std::vector<float> vecStds = { 1.0f, 1.0f, 1.0f });
    ~KeyframeLGTAInferer();

    int doInference(cv::Mat& src, std::string mode, std::vector<PeakInfo>& vPeaks);

    int doInference(cv::Mat& src, std::string mode, std::vector<float>& vecEDProbs, std::vector<float>& vecESProbs);

protected:
    int getSampledElements(int nElementCount, int nStride);

    int backboneInference(cv::Mat& src);

    int lgtaInference(cv::Mat& src, std::vector<float>& vecEDProbs, std::vector<float>& vecESProbs);

    std::vector<float*> getQueryFeat(int index) override;

    // 20250306更新：重写该方法，因为现在A4C和A2C是计算的最后一帧的概率，而不是PLAX那种计算中间帧
    int calcEachFrameAvgProb(std::vector<std::vector<float>>& vecTotalProbs, std::vector<float>& vecAvgProbs) override;
};

#endif // KEYFRAMEDETINFERER_H
