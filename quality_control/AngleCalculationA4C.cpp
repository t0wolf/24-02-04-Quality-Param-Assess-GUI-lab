#include "AngleCalculationA4C.h"



AngleCalculationA4C::AngleCalculationA4C(AngleCalculationParams params)
{
    // 切面类型名称
    assert(!params.viewname.empty()), "viewname is empty, please check your StructureDetectionParams!";
    this->viewname = params.viewname;
    // 切面分类模型文件名
    assert(!params.model_name.empty()), "model_name is empty, please check your StructureDetectionParams!";
    this->model_name = params.model_name;
    // 切面分类模型路径
    assert(!params.model_dir.empty()), "model_dir is empty, please check your StructureDetectionParams!";
    this->model_path = params.model_dir + this->model_name + ".engine";

    // 相应切面所包含的解剖结构
    assert(params.structures.size() >= 1), "structures is empty, please check your StructureDetectionParams!";
    this->structures = params.structures;
    // 目标检测的解剖结构表（检测框已剔除背景类，因此从0开始）
    //this->classnames.insert(this->classnames.begin(), "__background__");
    this->classnames = params.structures;
    // 各解剖结构在目标检测的编号
    for (int i = 0; i < this->classnames.size(); i++)
    {
        this->structure_idx_mapping[this->classnames[i]] = i;
    }
    // 二尖瓣的解剖结构名称符号 + 左心室的解剖结构名称符号
    //this->mv_name = params.mv_name;
    //this->lv_name = params.lv_name;
    this->mv_name = params.ac_structures[0];
    this->lv_name = params.ac_structures[1];


    // 预测时部分参数
    this->mMean = params.mMean;
    this->angle_threshold = params.angle_threshold;
    this->inputSize = cv::Size(params.inputSize, params.inputSize); // 模型输入图像尺寸（由于TensorRT特性，实际并未用到）
    this->mInputSize = this->inputSize;
    this->mOutputSize = this->inputSize;
    this->load_model();
}

AngleCalculationA4C::~AngleCalculationA4C()
{
    // 释放vector
    this->mMean.clear();

    // 释放模型
    delete this->model;
}


void
AngleCalculationA4C::load_model()
{
    std::fstream _file;
    _file.open(this->model_path.c_str(), std::ios::in);
    assert(_file), "Cannot find angle_calculation model!";
    _file.close();

    this->model = new AngleInferA4C(this->model_path, this->mInputSize, this->mOutputSize);
}


////////////////////////////
//  line 85-88,121-124 存疑
////////////////////////////

float
AngleCalculationA4C::predict(std::vector<cv::Mat> frames,
    std::vector<int> keyframes,
    f_structures frame_structures)
{
    cv::Point a;
    std::vector<float> angles;
    // 从关键帧中抽帧

    std::vector<std::vector<cv::Rect>> rects;
    std::vector<std::vector<int>> labels;
    std::vector<std::vector<float>> probs;
    std::vector<cv::Mat> frames_crop;

    // ---------------------------------------- 这部分存疑 ---------------------------------------- //
    //int lv_idx = this->structure_idx_mapping[this->mv_name];
    //int mv_idx = this->structure_idx_mapping[this->lv_name];
    int lv_idx = this->structure_idx_mapping[this->lv_name];
    int mv_idx = this->structure_idx_mapping[this->mv_name];
    // -------------------------------------------------------------------------------------------- //
   // *遍历关键帧索引，对于每个关键帧以及其之后的5帧（以防溢出），提取包含LV和MV结构的帧信息，
     //   * 包括矩形区域、标签、概率和裁剪后的图像，并存储到对应的向量中。
    // 当存在关键帧时，依据关键帧读取，否则直接取所有帧并求平均
    for (auto keyindex : keyframes)
    {
        for (int i = 0; i < 5; i++)//关键帧及其接下来的五帧都需要
        {
            // 以防溢出
            if (keyindex + i >= frames.size())
            {
                break;
            }
            // 存在并非所有帧都有对应结构的情况
            if (frame_structures.find(int(keyindex+i)) != frame_structures.end())
            {
                std::vector<int> label = frame_structures[keyindex+i].labels;
                // LV 与 MV需要同时存在
                //// 在vector中找到对应对象的索引（取最小值）
                if ((get_vector_idx_int(label, lv_idx) > 0) && (get_vector_idx_int(label, mv_idx) > 0))
                {
                    rects.push_back(frame_structures[keyindex + i].rects);
                    labels.push_back(frame_structures[keyindex + i].labels);
                    probs.push_back(frame_structures[keyindex + i].probs);
                    frames_crop.push_back(frames[keyindex + i].clone());//frame就是图像
                }
            }
        }
        if (labels.empty())
            continue;

        for (int i = 0; i < labels.size(); i++)
        {
            // ---------------------------------------- 这部分存疑 ---------------------------------------- //
            //int rectidx_mv = get_vector_idx_int(labels[i], lv_idx);
            //int rectidx_lv = get_vector_idx_int(labels[i], mv_idx);
            int rectidx_mv = get_vector_idx_int(labels[i], mv_idx);//貌似是得到那个框子的id
            int rectidx_lv = get_vector_idx_int(labels[i], lv_idx);
            // -------------------------------------------------------------------------------------------- //
            cv::Rect rect_lv = rects[i][rectidx_lv];
            cv::Rect rect_mv = rects[i][rectidx_mv];
            cv::Point mv = cv::Point(int(rect_mv.x + rect_mv.width / 2), int(rect_mv.y + rect_mv.height / 2));   //二尖瓣目标检测框的中点
            cv::Mat now = frames_crop[i].clone();
            cv::Mat showw = frames_crop[i].clone();
            cv::Mat tempframe = now(rect_lv);//截取图像内容进行推理
            a = this->model->doInference(tempframe);
            a = cv::Point(a.x + rect_lv.x, a.y + rect_lv.y);
            angles.push_back(atan2((mv.x - a.x), (mv.y - a.y)) * 180 / CV_PI);
        }
        rects.clear();
        labels.clear();
        probs.clear();
        frames_crop.clear();
    }

    // 若前述流程中未能得到结果，或关键帧序列为空，则对每一帧执行类似过程
    if (angles.empty() || keyframes.empty())
    {
        for (int i = 0; i < frames.size(); i++)
        {
            // 存在并非所有帧都有对应结构的情况
            if (frame_structures.find(i) != frame_structures.end())
            {
                std::vector<int> label = frame_structures[i].labels;
                // LV 与 MV需要同时存在
                if ((get_vector_idx_int(label, lv_idx) > 0) && (get_vector_idx_int(label, mv_idx) > 0))
                {
                    rects.push_back(frame_structures[i].rects);
                    labels.push_back(frame_structures[i].labels);
                    probs.push_back(frame_structures[i].probs);
                    frames_crop.push_back(frames[i].clone());
                }
            }
        }
        // 对LV与MV同时存在的每一帧执行类似过程
        for (int i = 0; i < labels.size(); i++)
        {
            int rectidx_lv = get_vector_idx_int(labels[i], lv_idx);
            int rectidx_mv = get_vector_idx_int(labels[i], mv_idx);
            cv::Rect rect_lv = rects[i][rectidx_lv];
            cv::Rect rect_mv = rects[i][rectidx_mv];
            cv::Point mv = cv::Point(int(rect_mv.x + rect_mv.width / 2), int(rect_mv.y + rect_mv.height / 2));
            cv::Mat now = frames_crop[i].clone();
            cv::Mat tempframe = now(rect_lv);
            a = this->model->doInference(tempframe);
            a = cv::Point(a.x + rect_lv.x, a.y + rect_lv.y);
            
            angles.push_back(atan2((mv.x - a.x), (mv.y - a.y)) * 180 / CV_PI);
        }
        rects.clear();
        labels.clear();
        probs.clear();
        frames_crop.clear();
    }

    float angle = std::accumulate(std::begin(angles), std::end(angles), 0.0) / angles.size();//求均值
    
    float angle_sorce = angle > this->angle_threshold ? 0.0 : 1.0;

    return angle_sorce;
}
