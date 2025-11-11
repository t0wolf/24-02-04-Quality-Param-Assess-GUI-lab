#include "StructureDetection.h"


StructureDetection::StructureDetection(StructureDetectionParams params)
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
    assert(params.structures.size()>=1), "structures is empty, please check your StructureDetectionParams!";
    this->structures = params.structures;
    // 目标检测的解剖结构表（检测框已剔除背景类，因此从0开始）
    //this->classnames.insert(this->classnames.begin(), "__background__");
    this->classnames = params.structures;

    // 各解剖结构的score阈值
    assert(!params.score_threshold.empty()), "score_threshold is empty, please check your StructureDetectionParams!";
    for (int i = 0; i < this->structures.size(); i++ )
    {
        this->score_threshold[this->structures[i]] = float(params.score_threshold[i]);
    }
    // 检测到解剖结构的帧数阈值
    assert(!params.frame_threshold.empty()), "frame_threshold is empty, please check your StructureDetectionParams!";
    for (int i = 0; i < this->structures.size(); i++)
    {
        this->frame_threshold[this->structures[i]] = int(params.frame_threshold[i]);
    }

    // 各解剖结构在目标检测的编号
    for (int i = 0; i < this->classnames.size(); i++)
    {
        this->idx_structure_mapping[i] = this->classnames[i];
    }

    // 预测时部分参数
    this->mMean = params.mMean;
    this->inputSize = cv::Size(params.inputSize, params.inputSize); // 模型输入图像尺寸（由于TensorRT特性，实际并未用到）
    this->load_model();
}

StructureDetection::~StructureDetection()
{
    // 释放vector
    std::vector<std::string>().swap(this->structures);
    std::vector<std::string>().swap(this->classnames);
    std::vector<float>().swap(this->mMean);

    // 释放模型
    delete this->model;
}


void
StructureDetection::load_model()
{
    std::fstream _file;
    _file.open(this->model_path.c_str(), std::ios::in);
    assert(_file), "Cannot find structure_detection model!";
    _file.close();

    this->model = new StructureInfer(this->model_path, this->inputSize, this->mMean, this->structures.size());
}


void
StructureDetection::predict(std::vector<cv::Mat> frames,
    s_frames& structure_frames,
    f_structures& frame_structures)
{
    this->model->doInference(frames, this->idx_structure_mapping, structure_frames, frame_structures);
}