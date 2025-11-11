#include "EchoQualityAssessmentPSAXA.h"

EchoQualityAssessmentPSAXA::EchoQualityAssessmentPSAXA(EchoQualityAssessmentParams params)
{
    // -------------------------------- 先验知识参数初始化 -------------------------------- //
    this->mParams = &params;
    // 切面类型名称
    //assert(!this->mParams->viewname.empty()), "viewname is empty, please check your EchoQualityAssessmentParams!";
    if (this->mParams->viewname.size() > 1) {
        this->viewname = this->mParams->viewname;
    }
    // 相应切面所包含的解剖结构
    //assert(!this->mParams->structures.empty()), "structures is empty, please check your EchoQualityAssessmentParams!";
    if (this->mParams->structures.size() > 1) {
        this->structures = this->mParams->structures;
    }
    this->classnames = this->structures;
    // 各解剖结构在目标检测的编号
    for (int i = 0; i < this->classnames.size(); i++)
    {
        this->structure_idx_mapping[this->classnames[i]] = i;
    }
    // 切面缩放比例基准解剖结构
    //assert(!this->mParams->scaling_structure.empty()), "scaling_structure is empty, please check your EchoQualityAssessmentParams!";
    if (this->mParams->scaling_structure.size() >= 1)
    {
        this->scaling_structure = this->mParams->scaling_structure;
    }
    //
    if (this->mParams->mMean.size() >= 1)
    {
        this->mMean = this->mParams->mMean;
    }

    // -------------------------------- 得分项初始化 -------------------------------- //
    // 各得分项相关参数
    this->score_weighted["structure"] = 1.0f;
    this->score_weighted["gain"] = 1.0f;
    this->score_weighted["scaling"] = 1.0f;
    this->score_weighted["integrity"] = 4.0f;

    // 解剖结构
    this->structure_score_structures.routine_must = {};  // 常规必须项结构
    this->structure_score_structures.routine_optional = { "RV" };                               // 常规可选项结构
    this->structure_score_structures.unroutine_optional = { "integrity" };                  // 非常规项结构
    this->structure_score_structures.unroutine_optional_params = {};      // 非常规项参数
    this->structure_score_weight.routine_must = 0.0f;                               // 常规必须项结构分数权重
    this->structure_score_weight.routine_optional = { -0.5f };                             // 常规可选项结构分数权重
    this->structure_score_weight.unroutine_optional = { 4.0f };               // 非常规项结构分数权重
    assert(this->structure_score_structures.routine_optional.size() ==
        this->structure_score_weight.routine_optional.size()), "please check params!";
    //assert(this->structure_score_structures.unroutine_optional_params.size() ==
    //    this->structure_score_weight.unroutine_optional.size()), "please check params!";
    assert(this->structure_score_structures.unroutine_optional.size() ==
        this->structure_score_weight.unroutine_optional.size()), "please check params!";

    // 缩放比例
    if (!params.scaling_structure.empty())
        this->scaling_structure = params.scaling_structure;
    if (!params.scaling_range.empty())
        this->scaling_range = params.scaling_range;

    // -------------------------------- 载入各模型 -------------------------------- //
    this->mParams->sdetection_params.viewname = "psaxa";
    this->mParams->sdetection_params.structures = this->structures;
    this->mParams->sdetection_params.classnames = this->classnames;
    this->mParams->sdetection_params.model_name = "structure_detection_psaxa";
   // this->mParams->sdetection_params.model_dir = "../../extern/structure_detection/";
    this->mParams->sdetection_params.model_dir = "D:\\Resources\\20240221\\quality_control_models\\structure_detection_models\\";
    this->mParams->sdetection_params.score_threshold = { 0.5f, 0.5f };
    this->mParams->sdetection_params.frame_threshold = { 1, 1 };
    this->mParams->sdetection_params.structures = this->structures;
    this->mParams->sdetection_params.mMean = this->mMean;

    this->mParams->gain_params.viewname = "psaxa";
    this->mParams->gain_params.frame_threshold = 0.1f;
    this->mParams->gain_params.models_dir = "D:\\Resources\\20240221\\quality_control_models\\gain_classification\\";
    //////////////// 20230507更新信息：由于弃用局部直方图+随机森林的增益分类方法，重写GainClassification，该部分弃用
    //this->mParams->gain_params.view_structures = this->structures;
    //this->mParams->gain_params.gain_structures = { "HEART" };
    //this->mParams->gain_params.gain_threshold = { 0.5f, 0.5f };
    //this->mParams->gain_params.frame_threshold = { 0.05f, 0.05f };

    this->frame_threshold = this->mParams->sdetection_params.frame_threshold;
    this->score_threshold = this->mParams->sdetection_params.score_threshold;

    load_integrity_model();
    load_structure_model();                                // 解剖结构检测模型载入
    load_gain_model();                                     // 图像增益模型载入
}

EchoQualityAssessmentPSAXA::~EchoQualityAssessmentPSAXA()
{

    // 释放私有变量内存，以防内存泄漏std::string
    std::vector<std::string>().swap(this->classnames);
    std::vector<std::string>().swap(this->structures);
    std::vector<float>().swap(this->mMean);
    std::vector<float>().swap(this->scaling_range);
    std::vector<float>().swap(this->score_threshold);
    std::vector<int>().swap(this->frame_threshold);
    std::vector<int>().swap(this->keyframes);

    //s_i_map().swap(this->structure_idx_mapping);
    //s_f_map().swap(this->score_weighted);
    //s_frames().swap(this->structure_frames);
    //f_structures().swap(this->frame_structures);

    // 释放模型
    delete this->structure_detection;
    delete this->gain_classification;
    delete this->integrity_classification;
}

void
EchoQualityAssessmentPSAXA::load_structure_model()
{
    //this->mParams->sdetection_params.viewname = "a4c";
    //this->mParams->sdetection_params.structures = this->structures;
    //this->mParams->sdetection_params.classnames = this->classnames;
    //this->mParams->sdetection_params.model_name = "structure_detection_a4c";
    //this->mParams->sdetection_params.model_dir = "../../extern/structure_detection/";
    //this->mParams->sdetection_params.score_threshold = { 0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5 };
    //this->mParams->sdetection_params.frame_threshold = { 1,1,1,1,1,1,1,1,1 };
    //this->mParams->sdetection_params.structures = this->structures;
    //this->mParams->sdetection_params.mMean = this->mMean;

    this->structure_detection = new StructureDetection(this->mParams->sdetection_params);
}

void
EchoQualityAssessmentPSAXA::load_integrity_model()
{
  //  std::string enginePath = "../../extern/integrity_classification/intima_classification.engine";
    std::string enginePath = "D:/Resources/20240221/quality_control_models/integrity_models/psaxa_integrity.engine";

    this->integrity_classification = new IntegrityClassification(enginePath);

}

void
EchoQualityAssessmentPSAXA::load_gain_model()
{
    //this->mParams->gain_params.viewname = "a4c";
    //this->mParams->gain_params.view_structures = this->structures;
    //this->mParams->gain_params.gain_structures = { "RV" };
    //this->mParams->gain_params.gain_threshold = { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
    //this->mParams->gain_params.frame_threshold = { 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f };

    this->gain_classification = new GainClassification(this->mParams->gain_params);
}

float EchoQualityAssessmentPSAXA::structure_judgment(s_frames structure_frames,
    std::vector<int> keyframes,
    int frame_nums,
    std::vector<float>& grades)
{
    // 根据是否检测到对应的解剖结构，添加1或0
    s_frames structure_preds;
    for (auto it : structure_frames) {
        std::vector<int> frame_in_res;
        for (int i = 0; i < frame_nums; i++) {
            int temp = get_vector_idx_int(it.second, i) > 0 ? 1 : 0;
            frame_in_res.push_back(temp);
        }
        structure_preds[it.first] = frame_in_res;
    }


    // 常规结构（必须得分项）合并统计
    //std::vector<std::string> routine_must = this->structure_score_structures.routine_must;
    //float routine_must_flag = 0.0;
    //for (std::string structure : routine_must)
    //{
    //    int structure_idx = this->structure_idx_mapping[structure];
    //    int temp = int(std::accumulate(structure_preds[structure].begin(), structure_preds[structure].end(), 0.0));
    //    float pred_structure = temp >= this->frame_threshold[structure_idx] ? 1.0 : 0.0;
    //    routine_must_flag += pred_structure;
    //}

    //// 必须得分项需要每一项都得分
    //float routine_must_sorce = size_t(routine_must_flag) == routine_must.size() ? 1.0 : 0.0;
    //this->structure_scores.routine_must = routine_must_sorce * this->structure_score_weight.routine_must;

    //grades.push_back(structure_scores.routine_must);

    // 常规结构（可选得分项）分别统计
    std::vector<std::string> routine_optional = this->structure_score_structures.routine_optional;
    if (!routine_optional.empty())
    {
        for (int i = 0; i < routine_optional.size(); i++)
        {
            std::string structure = routine_optional[i];
            int structure_idx = this->structure_idx_mapping[structure];
            int temp = int(std::accumulate(structure_preds[structure].begin(), structure_preds[structure].end(), 0.0));
            float pred_structure = temp >= this->frame_threshold[structure_idx] ? 1.0 : 0.0;
            pred_structure *= this->structure_score_weight.routine_optional[i];
            this->structure_scores.routine_optional.push_back(pred_structure);
            grades.push_back(pred_structure);
        }
        
    }

    // 非常规结构单独统计
   /* std::vector<std::string> unroutine_optional = this->structure_score_structures.unroutine_optional;
    std::vector<float> unroutine_optional_params = this->structure_score_structures.unroutine_optional_params;
    if (!unroutine_optional.empty())
    {
        for (int i = 0; i < unroutine_optional.size(); i++)
        {
            std::string structure = unroutine_optional[i];
            int structure_idx = this->structure_idx_mapping[structure];
            std::vector<int> structure_preds_sub = structure_preds[structure];
            std::vector<int>::const_iterator preds_first = structure_preds_sub.begin();
            if (keyframes.size() > 1)
            {
                float pred_structure = 0.0;
                for (int j = 1; j < keyframes.size(); j++)
                {
                    int keyframe_s = keyframes[j - 1];
                    int keyframe_e = keyframes[j];
                    std::vector<int> subpreds;
                    subpreds.assign(preds_first + keyframe_s, preds_first + keyframe_e);
                    float temp = float(std::accumulate(subpreds.begin(), subpreds.end(), 0.0));
                    if (temp >= subpreds.size() * unroutine_optional_params[i])
                    {
                        pred_structure = 1.0;
                        break;
                    }
                }
                pred_structure *= this->structure_score_weight.unroutine_optional[i];
                this->structure_scores.unroutine_optional.push_back(pred_structure);
    
            }
        }
    }*/

    this->structure_score = 0.0f;

    this->structure_score += this->structure_scores.routine_must;

    if (!this->structure_scores.routine_optional.empty())
    {
        this->structure_score += float(std::accumulate(
            structure_scores.routine_optional.begin(), structure_scores.routine_optional.end(), 0.0));
    }
    if (!this->structure_scores.unroutine_optional.empty())
    {
        this->structure_score += float(std::accumulate(
            structure_scores.unroutine_optional.begin(), structure_scores.unroutine_optional.end(), 0.0));
    }

    return this->structure_score;
}

float
EchoQualityAssessmentPSAXA::scaling_judgment(s_frames structure_frames,
    f_structures frame_structures,
    std::vector<float> radius)
{
    std::vector<int> frame_idxs = structure_frames[this->scaling_structure];
    int structure_idx = this->structure_idx_mapping[this->scaling_structure];

    std::vector<float> heart_heights;
    for (int frame_idx: frame_idxs)
    {
        int box_idx = get_vector_idx_int(frame_structures[frame_idx].labels, structure_idx);
        cv::Rect boxes = frame_structures[frame_idx].rects[box_idx];
        heart_heights.push_back(float(boxes.height));
    }

    std::vector<float>::iterator biggest = std::max_element(heart_heights.begin(), heart_heights.end());
    std::vector<float>::iterator miniest = std::min_element(heart_heights.begin(), heart_heights.end());
    
    float heart_height_sum = float(std::accumulate(heart_heights.begin(), heart_heights.end(), 0.0));
    float heart_height_max = *biggest;
    float heart_height_min = *miniest;

    float real_radius = 0.0f;
    if (radius.size() == 1)
    {
        real_radius = radius[0];
    }
    else
    {
        real_radius = abs(radius[1] - radius[0]);
    }
    
    float scaling_min = heart_height_min / real_radius;
    float scaling_max = heart_height_max / real_radius;
    float scaling_mean = heart_height_sum / heart_heights.size() / real_radius;

    bool mean_flag = (scaling_mean >= this->scaling_range[0] - 0.01) && (scaling_mean <= this->scaling_range[1] + 0.01);
    bool max_min_flag = (scaling_min >= this->scaling_range[0] - 0.1) && (scaling_max <= this->scaling_range[1] + 0.1);
    this->scaling_score = (mean_flag && max_min_flag) ? 1.0 : 0.0;

    return this->scaling_score;
}


float 
EchoQualityAssessmentPSAXA::gain_judgment(
    std::vector<cv::Mat> frames)
{
    //////////////// 20230507更新信息：由于弃用局部直方图+随机森林的增益分类方法，重写GainClassification，该部分弃用
    //this->gain_score = this->gain_classification->predict(structure_frames, frame_structures, frames, roi_mask);
    this->gain_score = this->gain_classification->predict(frames);

    return this->gain_score;
}

s_f_map
EchoQualityAssessmentPSAXA::predict(
    std::vector<cv::Mat> frames,  std::vector<float> radius, singleResult& results)
{
    // 增益得分
////////////////// 20230507更新信息：由于弃用局部直方图+随机森林的增益分类方法，重写GainClassification，该部分弃用
//this->gain_score = this->gain_judgment(this->structure_frames, this->frame_structures, frames, roi_mask);
    this->gain_score = this->gain_judgment(frames);
    float gain_score_out = this->gain_score * this->score_weighted["gain"];
    std::cout << "gain_score" << gain_score_out << std::endl;
    results.grades.push_back(gain_score_out);
    // 解剖结构检测
    this->structure_detection->predict(frames, this->structure_frames, this->frame_structures);

    if (this->structure_frames.empty())
        return s_f_map();

    // 心肌完整性检测
    int integrityStatus = this->integrity_classification->inference(frames);
   // int b =this->
    //int integrityStatus = classResults.first;
    // 缩放比例得分
    float scaling_score_out = 0.0f;
    if (this->structure_frames.find(scaling_structure) != this->structure_frames.end())
    {
        if (!this->structure_frames[scaling_structure].empty())
        {
            this->scaling_score = this->scaling_judgment(this->structure_frames, this->frame_structures, radius);
            results.grades.push_back(this->scaling_score);
        }
    }

    // 解剖结构得分
    this->structure_score = this->structure_judgment(this->structure_frames, this->keyframes, int(frames.size()), results.grades);

    this->total_score = 0.0f;
    this->total_score += this->gain_score * this->score_weighted["gain"];
    this->total_score += this->scaling_score * this->score_weighted["scaling"];
    this->total_score += this->structure_score * this->score_weighted["structure"];

    //if (integrityStatus == 0)
    //    this->integrity_score = 4.0f;
    //else
    //    this->integrity_score = 2.0f;
    switch (integrityStatus)
    {
    case(0):
        this->integrity_score = 4.0f;
        break;
    case(1):
        this->integrity_score = 3.0f;
        break;
    case(2):
        this->integrity_score = 2.0f;
        break;
    case(3):
        this->integrity_score = 0.0f;
        break;
    default:
        this->integrity_score = 0.0f;
        break;
    }

    s_f_map PSAXA_MAP;
    PSAXA_MAP["gain_score"] = this->gain_score;
    PSAXA_MAP["structure_score"] = this->structure_score;
    PSAXA_MAP["scaling_score"] = this->scaling_score;
    PSAXA_MAP["integrityStatus"] = this->integrity_score;
    return PSAXA_MAP;
}