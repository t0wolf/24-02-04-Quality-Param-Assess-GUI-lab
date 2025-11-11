//
// Created by 单淳劼，曾子炀；成汉林重构
//
#include "cuda_runtime_api.h"
#include <algorithm>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
//#include "logging.h"
//#include "BatchStream.h"
//#include "EntropyCalibrator.h"
//#include "argsParser.h"
//#include "buffers.h"
//#include "common.h"
#include "logger.h"
#include "NvCaffeParser.h"

#include "cplusfuncs.h"
#include "typemappings.h"

using namespace nvinfer1;
//using namespace sample;

class StructureInfer
{
public:
    StructureInfer(std::string& trtEnginePath, cv::Size& inputSize, std::vector<float> Mean, int objectNum);

    ~StructureInfer();

    void doInference(std::vector<cv::Mat> frames, i_s_map idx_structure_mapping,
        s_frames& structure_frames, f_structures& frame_structures);

private:
    int initialize();

    //void findfirstProb(const std::vector<Object>& vObjects, std::vector<int>& picked, float nms_threshold);

    void preprocess(cv::Mat& src, cv::Mat& dst);

    cv::Mat doSingleInfer(cv::Mat& src, std::vector<cv::Rect>& rect, std::vector<int>& label, std::vector<float>& prob);

    void blobFromImage(cv::Mat& image);

    void decodeOutputs(std::vector<cv::Rect>& rect, std::vector<int>& label, std::vector<float>& prob);

    void generate_proposals(float probThresh, std::vector<std::vector<object_box>>& objects);

    void qsort_descent_inplace(std::vector<object_box>& vObjects, int left, int right);

    std::string mEnginePath;
    logger::Logger mGLogger;
    IRuntime* mRuntime;
    ICudaEngine* mEngine;
    IExecutionContext* mContext;
    cudaStream_t mStream;

    cv::Size mcvInputSize;
    cv::Size mcvOriginSize;
    int mSingleVideoFrameCount;

    int mInputSize;
    int mInputIndex;
    int mobjectNum;

    int mBoxesSize;
    int mLabelsSize;
    int mScoresSize;
    int mLabelsIndex;
    int mBoxesIndex;
    int mScoresIndex;

    float* mBlob;
    float* mLabels;
    float* mBoxes;
    float* mScores;
    void* mBuffers[4];

    std::vector<float> vMeans = { 123.0f, 117.0f, 104.0f };

    static std::pair<int, float> argmax(std::vector<float>& vProb);

    static void checkStatus(cudaError status)
    {
        do
        {
            auto ret = (status);
            if (ret != 0)
            {
                std::cerr << "Cuda failure: " << ret << std::endl;
                abort();
            }
        } while (0);
    }

    void qsort_descent_inplace(std::vector<object_box>& objects);

    static void nms_sorted_bboxes(const std::vector<object_box>& vObjects, std::vector<int>& picked, float nms_threshold);

    static inline float intersection_area(const object_box& a, const object_box& b)
    {
        cv::Rect_<float> inter = a.rect & b.rect;
        return inter.area();
    }
};

