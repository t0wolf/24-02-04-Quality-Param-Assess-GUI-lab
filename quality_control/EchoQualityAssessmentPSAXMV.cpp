#include "EchoQualityAssessmentPSAXMV.h"

EchoQualityAssessmentPSAXMV::EchoQualityAssessmentPSAXMV(EchoQualityAssessmentParams params)
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

    //// 解剖结构
    // 常规必须项：集合内元素均需被检测到，缺一不可的解剖结构
    // 常规可选项：检测到即得分的解剖结构
    // 非常规项：  需要单独处理的解剖结构，如A4C“二尖瓣显示清晰”除各帧检测结果，还需要结合心动周期

    /////////////////// 2023.12.03根据单淳劼师弟反馈，注释并更新
    //this->structure_score_structures.routine_must = { "MV", "LVB" };  // 常规必须项结构
    //this->structure_score_structures.routine_optional = { "LVS", "LVF" };      // 常规可选项结构
    //this->structure_score_structures.unroutine_optional = { "MV_judge" };     // 非常规项结构
    this->structure_score_structures.routine_must = { "LVB" };  // 常规必须项结构
    this->structure_score_structures.routine_optional = { "LVS", "LVF" };      // 常规可选项结构
    this->structure_score_structures.unroutine_optional = { "MV" };     // 非常规项结构

    this->structure_score_structures.unroutine_optional_params = { 1.0f };   // 非常规项参数
    this->structure_score_weight.routine_must = 5.0f;                        // 常规必须项结构分数权重
    this->structure_score_weight.routine_optional = { 1.0f, 1.0f };          // 常规可选项结构分数权重
    this->structure_score_weight.unroutine_optional = { 1.0f };              // 非常规项结构分数权重
    assert(this->structure_score_structures.routine_optional.size() ==
        this->structure_score_weight.routine_optional.size()), "please check params!";
    assert(this->structure_score_structures.unroutine_optional_params.size() ==
        this->structure_score_weight.unroutine_optional.size()), "please check params!";

    // 缩放比例
    if (!params.scaling_structure.empty())
        this->scaling_structure = params.scaling_structure;
    if (!params.scaling_range.empty())
        this->scaling_range = params.scaling_range;

    // -------------------------------- 载入各模型 -------------------------------- //
    this->mParams->sdetection_params.viewname = "psaxmv";
    this->mParams->sdetection_params.structures = this->structures;
    this->mParams->sdetection_params.classnames = this->classnames;
    this->mParams->sdetection_params.model_name = "structure_detection_psaxmv";
  //  this->mParams->sdetection_params.model_dir = "../../extern/structure_detection/";
    this->mParams->sdetection_params.model_dir = "D:\\Resources\\20240221\\quality_control_models\\structure_detection_models\\";
    this->mParams->sdetection_params.score_threshold = { 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.2f };
    this->mParams->sdetection_params.frame_threshold = { 1, 1, 1, 1, 1, 1, 1, 1 };
    //this->scaling_range = { 1.0f / 2.0f, 4.0f / 5.0f };
    this->mParams->sdetection_params.structures = this->structures;
    this->mParams->sdetection_params.mMean = this->mMean;
    this->mParams->gain_params.viewname = "psaxmv";
    this->mParams->gain_params.frame_threshold = 0.2f;
    this->mParams->gain_params.models_dir = "D:\\Resources\\20240221\\quality_control_models\\gain_classification\\";


    //////////////// 20230507更新信息：由于弃用局部直方图+随机森林的增益分类方法，重写GainClassification，该部分弃用
    //this->mParams->gain_params.view_structures = this->structures;
    //this->mParams->gain_params.gain_structures = { "HEART" };
    //this->mParams->gain_params.gain_threshold = { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
    //this->mParams->gain_params.frame_threshold = { 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f };

    this->frame_threshold = this->mParams->sdetection_params.frame_threshold;
    this->score_threshold = this->mParams->sdetection_params.score_threshold;

    load_structure_model();                                // 解剖结构检测模型载入
    load_gain_model();
}

EchoQualityAssessmentPSAXMV::~EchoQualityAssessmentPSAXMV()
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
}

void
EchoQualityAssessmentPSAXMV::load_structure_model()
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
EchoQualityAssessmentPSAXMV::load_gain_model()
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

    this->gain_classification = new GainClassification(this->mParams->gain_params);
}


float EchoQualityAssessmentPSAXMV::structure_judgment(s_frames structure_frames,
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
    std::vector<std::string> routine_must = this->structure_score_structures.routine_must;
    float routine_must_flag = 0.0;
    for (std::string structure : routine_must)
    {
        int structure_idx = this->structure_idx_mapping[structure];
        int temp = int(std::accumulate(structure_preds[structure].begin(), structure_preds[structure].end(), 0.0));
        float pred_structure = temp >= this->frame_threshold[structure_idx] ? 1.0 : 0.0;
        routine_must_flag += pred_structure;
    }

    // 必须得分项需要每一项都得分
    float routine_must_sorce = size_t(routine_must_flag) == routine_must.size() ? 1.0 : 0.0;
    this->structure_scores.routine_must = routine_must_sorce * this->structure_score_weight.routine_must;
    float temp_score = this->structure_scores.routine_must;
    grades.push_back(temp_score);

    // 常规结构（可选得分项）分别统计
    //std::vector<int> vLVFExist;
    //std::vector<int> vLVSExist;
    //psaxmv_lvf_judge(vLVFExist, vLVSExist);

    std::vector<std::string> routine_optional = this->structure_score_structures.routine_optional;
    if (!routine_optional.empty())
    {
        for (int i = 0; i < routine_optional.size(); i++)
        {
            std::string structure = routine_optional[i];
            //float angle = *std::max_element(vAngle.begin(), vAngle.end());
            //std::string newStructure = "";
            //if (angle >= 30.0f)
            //{
            //    if (structure == "LVS")
            //        newStructure = "LVB";

            //    else if (structure == "LVF")
            //        newStructure = "LVS";
            //    else
            //        newStructure = structure;
            //}
            //else
            //    newStructure = structure;


            int structure_idx = this->structure_idx_mapping[structure];
            int temp = int(std::accumulate(structure_preds[structure].begin(), structure_preds[structure].end(), 0.0));

            float pred_structure = temp >= this->frame_threshold[structure_idx] ? 1.0 : 0.0;
            pred_structure *= this->structure_score_weight.routine_optional[i];

            float temp_score_temp = pred_structure;
            this->structure_scores.routine_optional.push_back(pred_structure);
            grades.push_back(temp_score_temp);
        }
    }

    // 非常规结构单独统计
    //auto vCircleJudgeResults = this->psaxmv_lv_judgement();
    //auto iter = std::find(vCircleJudgeResults.begin(), vCircleJudgeResults.end(), 1);
    //if (iter != vCircleJudgeResults.end())
    //    this->structure_score += 1.0f;
    auto vCircleJudgeResults = this->psaxmv_lv_judgement();
    auto iter = std::find(vCircleJudgeResults.begin(), vCircleJudgeResults.end(), 1);
    float temp_score2 = (iter != vCircleJudgeResults.end()) ? 1.0 : 0.0;
    temp_score2 *= this->structure_score_weight.unroutine_optional[0];
    this->structure_scores.unroutine_optional.push_back(temp_score2);
    grades.push_back(temp_score2);

    this->structure_score = 0.0f;

    this->structure_score += this->structure_scores.routine_must;

    if (!this->structure_scores.routine_optional.empty())
    {
        this->structure_score += float(std::accumulate(
            this->structure_scores.routine_optional.begin(), this->structure_scores.routine_optional.end(), 0.0));
    }
    if (!this->structure_scores.unroutine_optional.empty())
    {
        this->structure_score += float(std::accumulate(
            this->structure_scores.unroutine_optional.begin(), this->structure_scores.unroutine_optional.end(), 0.0));
    }

    return this->structure_score;
}

//void EchoQualityAssessmentPSAXMV::psaxmv_lvf_judge(std::vector<int>& vLVFExist, std::vector<int>& vLVSExist)
//{
//    for (auto frameStructure : this->frame_structures)
//    {
//        auto currLabels = frameStructure.second.labels;
//        auto currRects = frameStructure.second.rects;
//
//        int lvClsIdx = this->structure_idx_mapping["LV"];
//        int lvfClsIdx = this->structure_idx_mapping["LVF"];
//        auto lvIter = std::find(currLabels.begin(), currLabels.end(), lvClsIdx);
//        auto lvfIter = std::find(currLabels.begin(), currLabels.end(), lvfClsIdx);
//
//        if ((lvfIter != currLabels.end()) && (lvIter != currLabels.end()))
//        {
//            int lvIdx = std::distance(currLabels.begin(), lvIter);
//            int lvfIdx = std::distance(currLabels.begin(), lvfIter);
//
//            cv::Rect rectLv = currRects[lvIdx];
//            cv::Rect rectLvf = currRects[lvfIdx];
//
//            cv::Point2f ctrLvf(rectLvf.x + rectLvf.width / 2, rectLvf.y + rectLvf.height / 2);
//            cv::Point2f ctrLv(rectLv.x + rectLv.width / 2, rectLv.y + rectLv.height / 2);
//
//            float k1 = (ctrLv.y - ctrRv.y) / (ctrLv.x - ctrRv.x);
//            float k2 = 0.0f;
//
//            float theta = std::atan(std::abs((k2 - k1) / (1 + k1 * k2)));
//            float fAngle = theta * 180.0f / CV_PI;
//            vAngle.push_back(fAngle);
//        }
//        else
//            vAngle.push_back(-1.0f);
//    }
//}
float 
EchoQualityAssessmentPSAXMV::gain_judgment(
    std::vector<cv::Mat> frames)
{
    //////////////// 20230507更新信息：由于弃用局部直方图+随机森林的增益分类方法，重写GainClassification，该部分弃用
    //this->gain_score = this->gain_classification->predict(structure_frames, frame_structures, frames, roi_mask);
    this->gain_score = this->gain_classification->predict(frames);

    return this->gain_score;
}
float
EchoQualityAssessmentPSAXMV::scaling_judgment(s_frames structure_frames,
    f_structures frame_structures,
    std::vector<float> radius)
{
    std::vector<int> frame_idxs = structure_frames[this->scaling_structure];
    int structure_idx = this->structure_idx_mapping[this->scaling_structure];

    std::vector<float> heart_heights;
    for (int frame_idx : frame_idxs)
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





s_f_map
EchoQualityAssessmentPSAXMV::predict(
    std::vector<cv::Mat> frames, std::vector<float> radius, singleResult& results)
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

    // 缩放比例得分
    if (radius.empty() || this->structure_frames.empty() || !this->structure_frames.count(scaling_structure))
    {
        std::cout << "[E] Scaling judgment error, `radius` is empty!" << std::endl;
        this->scaling_score = 0.0f;
        results.grades.push_back(0.0f);
    }
    else
    {
        //this->scaling_score = this->s caling_judgment(this->structure_frames, this->frame_structures, radius);
        this->scaling_score = this->scaling_judgment(frames, this->structure_frames, this->frame_structures, radius);
        float scaling_score_out = this->scaling_score * this->score_weighted["scaling"];
        results.grades.push_back(scaling_score_out);
    }
    //this->scaling_score = this->scaling_judgment(this->structure_frames, this->frame_structures, radius);
    //results.grades.push_back(scaling_score);

    // 解剖结构得分
    this->structure_score = this->structure_judgment(this->structure_frames, this->keyframes, int(frames.size()), results.grades);

    s_f_map PSAXMV_MAP;
    PSAXMV_MAP["gain_score"] = this->gain_score;
    PSAXMV_MAP["structure_score"] = this->structure_score;
    PSAXMV_MAP["scaling_score"] = this->scaling_score;
    return PSAXMV_MAP;
}

bool
EchoQualityAssessmentPSAXMV::predict(
    std::vector<cv::Mat> frames, std::vector<cv::Mat>& vDrawnFrames, cv::Mat roi_mask, std::vector<float> radius, singleResult& results)
{
    // 解剖结构检测
    this->structure_detection->predict(frames, this->structure_frames, this->frame_structures);
    if (this->structure_frames.empty())
        return false;
   // std::cout<<"1"<<
    // 缩放比例得分
    if (radius.empty())
    {
        std::cout << "[E] Scaling judgment error, `radius` is empty!" << std::endl;
        this->scaling_score = 0.0f;
        results.grades.push_back(0.0f);
    }
    else
    {
        //this->scaling_score = this->scaling_judgment(this->structure_frames, this->frame_structures, radius);
        this->scaling_score = this->scaling_judgment(frames, vDrawnFrames, this->structure_frames, this->frame_structures, radius);
        float scaling_score_out = this->scaling_score * this->score_weighted["scaling"];
        results.grades.push_back(scaling_score_out);
    }
    //this->scaling_score = this->scaling_judgment(this->structure_frames, this->frame_structures, radius);
    //results.grades.push_back(scaling_score);

    // 解剖结构得分
    this->structure_score = this->structure_judgment(this->structure_frames, this->keyframes, int(frames.size()), results.grades);

    this->total_score = 0.0f;
    this->total_score += this->gain_score * this->score_weighted["gain"];
    this->total_score += this->scaling_score * this->score_weighted["scaling"];
    this->total_score += this->structure_score * this->score_weighted["structure"];
    results.totalGrade = total_score;
    this->is_standard = this->total_score >= this->total_score_th ? true : false;

    return this->is_standard;
}

std::vector<int> EchoQualityAssessmentPSAXMV::psaxmv_lv_judgement()
{
    std::vector<int> vResults;

    // 遍历所有的structures
    for (auto iter = this->frame_structures.begin(); iter != this->frame_structures.end(); iter++)
    {
        int frameIdx = iter->first;
        frame_object_boxes frameObj = iter->second;  // 当前帧的object boxes

        int lvClsIdx = this->structure_idx_mapping["LV"];
        auto lvIter = std::find(frameObj.labels.begin(), frameObj.labels.end(), lvClsIdx);  // 查找LV是否检测到
        if (lvIter != frameObj.labels.end())
        {
            int lvIdx = lvIter - frameObj.labels.begin();  // index of LV
            cv::Rect lvRect = frameObj.rects[lvIdx];  // rect of LV

            auto mvIter = std::find(frameObj.labels.begin(), frameObj.labels.end(), 5);  // if MV detected
            if (mvIter != frameObj.labels.end())
            {
                float fWidth = static_cast<float>(lvRect.width);
                float fHeight = static_cast<float>(lvRect.height);

                float ratio = fWidth / fHeight;

                if ((ratio >= this->ratioRange.first) && (ratio <= this->ratioRange.second))
                {
                    vResults.push_back(1);
                }
                else
                {
                    vResults.push_back(0);
                }
            }
        }
    }

    return vResults;
}

float
EchoQualityAssessmentPSAXMV::scaling_judgment(std::vector<cv::Mat>& vFrames,
    std::vector<cv::Mat>& vDrawnFrames,
    s_frames structure_frames,
    f_structures frame_structures,
    std::vector<float> radius)
{
    std::vector<int> frame_idxs = structure_frames[this->scaling_structure];
    int structure_idx = this->structure_idx_mapping[this->scaling_structure];

    std::vector<float> heart_heights;
    //cv::namedWindow("Scaling test");
    for (int frame_idx : frame_idxs)
    {
        int box_idx = get_vector_idx_int(frame_structures[frame_idx].labels, structure_idx);
        cv::Mat currFrame = vFrames[frame_idx];
        cv::Rect boxes = frame_structures[frame_idx].rects[box_idx];

        cv::rectangle(currFrame, boxes, cv::Scalar(0, 255, 0), 2);
        // 将带有绘制边界框的图像添加到输出向量
        vDrawnFrames.push_back(currFrame);
        //cv::imshow("Scaling test", currFrame);
        //cv::waitKey(30);
        heart_heights.push_back(float(boxes.height));//占比是看高度
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
        //real_radius = abs(radius[1] - radius[0]);
        float sum = std::accumulate(radius.begin(), radius.end(), 0.0f);
        real_radius = sum / static_cast<float>(radius.size());
    }

    float scaling_min = heart_height_min / real_radius;
    float scaling_max = heart_height_max / real_radius;
    float scaling_mean = heart_height_sum / heart_heights.size() / real_radius;

    bool mean_flag = (scaling_mean >= this->scaling_range[0] - 0.05) && (scaling_mean <= this->scaling_range[1] + 0.05);
    bool max_min_flag = (scaling_min >= this->scaling_range[0] - 0.05) && (scaling_max <= this->scaling_range[1] + 0.05);
    //scaling_range = { 1.0f / 2.0f, 4.0f / 5.0f };  // 相应切面缩放比例的合理范围
    this->scaling_score = (mean_flag && max_min_flag) ? 1.0 : 0.0;

    //cv::destroyWindow("Scaling test");
    return this->scaling_score;
}

float
EchoQualityAssessmentPSAXMV::scaling_judgment(std::vector<cv::Mat>& vFrames,
    s_frames structure_frames,
    f_structures frame_structures,
    std::vector<float> radius)
{
    std::vector<int> frame_idxs = structure_frames[this->scaling_structure];
    int structure_idx = this->structure_idx_mapping[this->scaling_structure];

    std::vector<float> heart_heights;
    for (int frame_idx : frame_idxs)
    {
        int box_idx = get_vector_idx_int(frame_structures[frame_idx].labels, structure_idx);
        cv::Mat currFrame = vFrames[frame_idx];
        cv::Rect boxes = frame_structures[frame_idx].rects[box_idx];

        //cv::rectangle(currFrame, boxes, cv::Scalar(0, 255, 0), 2);
        //vDrawnFrames.push_back(currFrame);
        //cv::imshow("Scaling test", currFrame);
        //cv::waitKey(30);
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
        //real_radius = abs(radius[1] - radius[0]);
        float sum = std::accumulate(radius.begin(), radius.end(), 0.0f);
        real_radius = sum / static_cast<float>(radius.size());
    }

    float scaling_min = heart_height_min / real_radius;
    float scaling_max = heart_height_max / real_radius;
    float scaling_mean = heart_height_sum / heart_heights.size() / real_radius;

    bool mean_flag = (scaling_mean >= this->scaling_range[0] - 0.05) && (scaling_mean <= this->scaling_range[1] + 0.05);
    bool max_min_flag = (scaling_min >= this->scaling_range[0] - 0.05) && (scaling_max <= this->scaling_range[1] + 0.05);
    this->scaling_score = (mean_flag && max_min_flag) ? 1.0 : 0.0;

    //cv::destroyWindow("Scaling test");
    return this->scaling_score;
}