#include "AngleCalculationPLAX.h"



AngleCalculationPLAX::AngleCalculationPLAX(AngleCalculationParams params)
{
    // 切面类型名称
    assert(!params.viewname.empty()), "viewname is empty, please check your StructureDetectionParams!";
    this->viewname = params.viewname;
    // 切面分类模型文件名
    //assert(!params.model_name_plax.empty()), "model_name is empty, please check your StructureDetectionParams!";
    this->model_name = params.model_name;
    // 切面分类模型路径
    assert(!params.model_dir.empty()), "model_dir is empty, please check your StructureDetectionParams!";
    this->model_path = params.model_dir + this->model_name + ".engine";

    // 相应切面所包含的解剖结构
    assert(params.structures.size() >= 1), "structures is empty, please check your StructureDetectionParams!";
    this->structures = params.structures;
    // 目标检测的解剖结构表（检测框已剔除背景类，因此从0开始）
    this->classnames = params.structures;
    // 各解剖结构在目标检测的编号
    for (int i = 0; i < this->classnames.size(); i++)
    {
        this->structure_idx_mapping[this->classnames[i]] = i;
    }
    // 结构名称符号
    this->lvlvot_name = params.ac_structures[0];


    // 预测时部分参数
    this->mMean = params.mMean;
    //this->angle_threshold = params.angle_threshold;
    this->inputSize = cv::Size(params.inputSize, params.inputSize); // 模型输入图像尺寸（由于TensorRT特性，实际并未用到）
    this->mInputSize = this->inputSize;
    this->mOutputSize = this->inputSize;
    this->load_model();
}

AngleCalculationPLAX::~AngleCalculationPLAX()
{
    // 释放vector
    this->mMean.clear();

    // 释放模型
    delete this->model;
}


void
AngleCalculationPLAX::load_model()
{
    std::fstream _file;
    _file.open(this->model_path.c_str(), std::ios::in);
    assert(_file), "Cannot find resnet50_plax model!";
    _file.close();

    this->model = new AngleInferPLAX(this->model_path, this->mInputSize, this->mOutputSize);
}


float
AngleCalculationPLAX::predict(std::vector<cv::Mat> frames,
    std::vector<int> keyframes,
    f_structures frame_structures)
{
    int class_result;
    //cv::Point a;
    std::vector<float> class_list;
    //std::vector<float> angles;
    // 从关键帧中抽帧

    std::vector<std::vector<cv::Rect>> rects;
    std::vector<std::vector<int>> labels;
    std::vector<std::vector<float>> probs;
    std::vector<cv::Mat> frames_crop;

    int lvlvot_idx = this->structure_idx_mapping[this->lvlvot_name];

    // 当存在关键帧时，依据关键帧读取，否则直接取所有帧并求平均
    for (auto keyindex : keyframes)
    {
        for (int i = 0; i < 5; i++)
        {
            // 以防溢出
            if (keyindex + i >= frames.size())
            {
                break;
            }
            // 存在并非所有帧都有对应结构的情况
            if (frame_structures.find(int(keyindex + i)) != frame_structures.end())
            {
                std::vector<int> label = frame_structures[keyindex + i].labels;
                // LVLVOT需要存在
                if (get_vector_idx_int(label, lvlvot_idx) > 0)
                {
                    rects.push_back(frame_structures[keyindex + i].rects);      //SSD的
                    labels.push_back(frame_structures[keyindex + i].labels);    //SSD的
                    probs.push_back(frame_structures[keyindex + i].probs);      //SSD的
                    frames_crop.push_back(frames[keyindex + i].clone());
                }
            }
        }
        if (labels.empty())
            continue;

        for (int i = 0; i < labels.size(); i++)
        {
            int rectidx_lvlvot = get_vector_idx_int(labels[i], lvlvot_idx);

            cv::Rect rect_lvlvot = rects[i][rectidx_lvlvot];
            cv::Mat now = frames_crop[i].clone();
            cv::Mat tempframe = now(rect_lvlvot);

            class_result = this->model->doInference(tempframe);     //二分类预测，pos:0, neg:1
            class_list.push_back(class_result);
        }
        std::cout << keyindex << std::endl;
        rects.clear();
        labels.clear();
        probs.clear();
        frames_crop.clear();
    }

    // 若前述流程中未能得到结果，或关键帧序列为空，则对每一帧执行类似过程
    if (class_list.empty() || keyframes.empty())
    {
        for (int i = 0; i < frames.size(); i++)
        {
            // 存在并非所有帧都有对应结构的情况
            if (frame_structures.find(i) != frame_structures.end())
            {
                std::vector<int> label = frame_structures[i].labels;
                // LVLVOT需要存在
                if (get_vector_idx_int(label, lvlvot_idx) > 0)
                {
                    rects.push_back(frame_structures[i].rects);
                    labels.push_back(frame_structures[i].labels);
                    probs.push_back(frame_structures[i].probs);
                    frames_crop.push_back(frames[i].clone());
                }
            }
        }
        // 对LVLVOT存在的每一帧执行类似过程
        for (int i = 0; i < labels.size(); i++)
        {
            int rectidx_lvlvot = get_vector_idx_int(labels[i], lvlvot_idx);
            cv::Rect rect_lvlvot = rects[i][rectidx_lvlvot];
            cv::Mat now = frames_crop[i].clone();
            cv::Mat tempframe = now(rect_lvlvot);
            class_result = this->model->doInference(tempframe);
            class_list.push_back(class_result);
        }
        rects.clear();
        labels.clear();
        probs.clear();
        frames_crop.clear();
    }

    float class_avg = std::accumulate(std::begin(class_list), std::end(class_list), 0.0) / class_list.size();

    float class_score = class_avg > this->class_threshold ? 0.0 : 1.0;  //最终分类结果

    return class_score;
}
