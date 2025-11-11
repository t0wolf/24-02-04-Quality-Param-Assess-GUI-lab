#include "GainClassification.h"



GainClassification::GainClassification(GainParams params)
{
    // -------------------------------- 先验知识参数初始化 -------------------------------- //
    // 切面类型名称
    this->viewname = params.viewname;
    // 切面类型名称符号与切面类型的字符映射
    for (int i = 0; i < this->name_and_mapname[0].size(); i++)
    {
        this->viewname_mapping[this->name_and_mapname[0][i]] = this->name_and_mapname[1][i];
    }
    // 增益分类子模型路径
    assert(!params.models_dir.empty()), "models_dir is empty, please check your GainParams!";
    this->models_dir = params.models_dir + this->viewname + "/";

    // dark/bright模型预测划分阈值（0：欠佳）
    //assert(!params.gain_threshold.empty()), "gain_threshold is empty, please check your GainParams!";
    this->gain_threshold = params.gain_threshold;
    // 图像增益欠佳帧数阈值
    //assert(!params.frame_threshold.empty()), "frame_threshold is empty, please check your GainParams!";
    this->frame_threshold = params.frame_threshold;

    // 预测时部分参数
    this->mMean = params.mMean;
    this->mStd = params.mStd;
    // this->mMeanThreshold = params.frame_mean_th;
    // this->mBrightRateThreshold = params.brightframe_rate_th;
    this->inputSize = cv::Size(params.inputSize, params.inputSize); // 模型输入图像尺寸（由于TensorRT特性，实际为固定值）

    this->load_model();
}


GainClassification::~GainClassification()
{
    // 释放私有变量内存，以防内存泄漏
    std::vector<float>().swap(this->mMean);
    s_s_map().swap(this->viewname_mapping);

    //// 增益分类模型
    //delete this->model_d;
    //delete this->model_b;
    delete this->model;
}


void GainClassification::load_model()
{
    std::fstream _file;

    // 载入模型
    std::string model_path = this->models_dir + "gain_" + this->viewname  + ".engine";

    _file.open(model_path.c_str(), std::ios::in);
    assert(_file), "Cannot find gain_classification model model!";
    _file.close();

    this->model = new GainInfer(model_path, this->inputSize, this->mMean, this->mStd);
     

    // 载入bright模型
    //std::string model_b_path = this->models_dir + "gain_" + this->viewname + this->modelname_s[1] + ".engine";

    //_file.open(model_b_path.c_str(), std::ios::in);
    //assert(_file), "Cannot find gain_classification model_bright model!";
    //_file.close();

    //this->model_b = new GainInfer(model_b_path, this->inputSize, this->mMean, this->mStd);
}


//int GainClassification::decide_model(std::vector<cv::Mat> frames, cv::Mat roi_mask)
//{
//    int tbright = 0;
//    // 计算视频中，较亮视频帧（均值>=40.0）的占比
//    for (int i = 0; i < frames.size(); i++)
//    {
//        int okk = 0, meanok = 0;
//        // 计算mask内的像素点的均值
//        for (auto it = roi_mask.begin<uchar>(), it0 = frames[i].begin<uchar>(); it != roi_mask.end<uchar>(); ++it, ++it0)
//        {
//            if (int((*it)) > 0)
//            {
//                meanok += int((*it0));
//                okk++;
//            }
//
//        }
//        meanok = meanok / okk; //一个视频单帧的mean
//        if (meanok > this->mMeanThreshold)
//        {
//            tbright += 1;
//        }
//    }
//
//    // dark模型返回0，bright模型返回1
//    float bright_percent = float(tbright) / frames.size();
//    int model_idx = bright_percent >= this->mBrightRateThreshold ? 1 : 0;
//
//    return model_idx;
//}


float GainClassification::predict(
    std::vector<cv::Mat> frames)
{
    //assert(!frames.empty()), "frames empty !";

    std::vector <cv::Mat> video_data_copy = frames;
    //cv::Mat roi_mask_copy = roi_mask.clone();

    // 根据视频帧中，较亮视频的占比，确定送入dark还是bright模型（0：dark；1：bright）
//    int model_idx = decide_model(video_data_copy, roi_mask_copy);

    // 利用dark/bright模型获取视频中每一帧的增益得分
    std::vector<int> frame_gain_scores;
    this->model->doInference(video_data_copy, frame_gain_scores);
    //if (model_idx == 0)
    //{
    //    this->model_d->doInference(video_data_copy, frame_gain_scores);
    //}
    //else
    //{
    //    this->model_b->doInference(video_data_copy, frame_gain_scores);
    //}

    // 根据模型每一帧的分数，是否大于增益良好帧数百分比，返回得分1或0
    float gain_sorce = 0.0;
    int sum = accumulate(frame_gain_scores.begin(), frame_gain_scores.end(), 0);
    gain_sorce = float(sum) / frame_gain_scores.size() >= frame_threshold ? 1.0 : 0.0;//是不是不晓得

    return gain_sorce;
}
