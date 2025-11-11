#include "KeyframeDetection.h"



KeyframeDetection::KeyframeDetection(KeyframeDetectionParams params)
{
    // 切面类型名称
    assert(!params.viewname.empty()), "viewname is empty, please check your KeyframeDetectionParams!";
    this->viewname = params.viewname;
    // 切面分类模型文件名
    assert(!params.model_name.empty()), "model_name is empty, please check your KeyframeDetectionParams!";
    this->model_name = params.model_name;
    // 切面分类模型路径
    assert(!params.model_dir.empty()), "model_dir is empty, please check your KeyframeDetectionParams!";
    this->model_path = params.model_dir + this->model_name + ".engine";

    // 预测时部分参数
    this->mMean = params.mMean;
    this->stride = params.stride;
    this->frame_num = params.frame_num;
    this->inputSize = cv::Size(params.inputSize, params.inputSize); // 模型输入图像尺寸（由于TensorRT特性，实际并未用到）
    this->load_model();
}

KeyframeDetection::~KeyframeDetection()
{
    // 释放vector
    std::vector<float>().swap(this->mMean);

    // 释放模型
    delete this->model;
}


void
KeyframeDetection::load_model()
{
    std::fstream _file;
    _file.open(this->model_path.c_str(), std::ios::in);
    assert(_file), "Cannot find keyframe_detection model!";
    _file.close();

    this->model = new KeyframeInfer(this->model_path, this->inputSize, this->frame_num, this->mMean);
}


std::vector<int>
KeyframeDetection::predict(std::vector<cv::Mat> frames)
{
    std::vector<int> keyframes;
    keyframes = this->model->doInference(frames, this->stride);

    return keyframes;
}
