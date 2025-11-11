#include "plax_params_assess.h"

PLAXParamsAssess::PLAXParamsAssess()
	:m_pIvsandpwAssesser(NULL)
	,m_pAvadAssesser(NULL)
	,m_pAsdAndsjdAssesser(NULL)
	,m_pAadAssesser(NULL)
	,m_pLaadAssesser(NULL)
    , m_pMultiLineAssesser(NULL)
    , m_pMultiLineAortaAssesser(NULL)
    , m_pPLAXAorticQCDetect(NULL)
{
	//m_pIvsandpwAssesser = new IVSAndPWAssess;

    std::string avadEnginePath = "D:/Resources/20240221/param_assess_models/AVAD_HRNet_0229_170.engine";
	m_pAvadAssesser = new AVADAssess(avadEnginePath);
    QtLogger::instance().logMessage("[I] PLAX AoD Model Loaded");

    //std::string asdAndsjdEnginePath = "C:/Project/QualityAssess_with_Qt/onnx/onnx_models/param_assess_models/AAO_0106.engine";
	//m_pAsdAndsjdAssesser = new ASDAndSJDAssess(asdAndsjdEnginePath);

    //std::string aadEnginePath = "C:/Project/QualityAssess_with_Qt/onnx/onnx_models/param_assess_models/AAO_0106.engine";
	//m_pAadAssesser = new AADAssess(aadEnginePath);
    std::string m_outpaintPath = "D:/Resources/20240221/param_assess_models/aotra_outpaint.engine";
    std::string m_multiLineMaskPath_Aorta = "D:/Resources/20240221/param_assess_models/aorta_maskline.engine";
    m_pMultiLineAortaAssesser = new MultiLineAssess_Aorta(m_outpaintPath, m_multiLineMaskPath_Aorta);
    QtLogger::instance().logMessage("[I] PLAX Aorta Model Loaded");
    //std::string laadEnginePathLa = "C:/Project/QualityAssess_with_Qt/onnx/onnx_models/param_assess_models/LAAD_LA_0220.engine";
    //std::string laadEnginePathAv = "C:/Project/QualityAssess_with_Qt/onnx/onnx_models/param_assess_models/LAAD_AV_0221.engine";
	//m_pLaadAssesser = new LAADAssess(laadEnginePathLa, laadEnginePathAv);

    //std::string multiStructPath = "C:/Project/QualityAssess_with_Qt/onnx/onnx_models/param_assess_models/IVS_PW_AO_LA.engine";
    //m_multiStructInferer = new MultiStructInferer(multiStructPath);

    std::string multiLinePath = "D:/Resources/20240221/param_assess_models/structure_params_line_detectpai_0719.engine";
    std::string detachTwoLinePath = "D:/Resources/20240221/param_assess_models/structure_params_line_detect_0827_2value.engine";
    std::string detachFourLinePath = "D:/Resources/20240221/param_assess_models/structure_params_line_detect_0827_4value.engine";
    std::string multiLineMaskPath = "D:/Resources/20240221/param_assess_models/structure_params_line_detect_1106_multi_value.engine";
    m_pMultiLineAssesser = new MultiLineAssess(multiLinePath, detachTwoLinePath, detachFourLinePath, multiLineMaskPath);
    QtLogger::instance().logMessage("[I] PLAX Multi-line Model Loaded");
}

PLAXParamsAssess::PLAXParamsAssess(ConfigParse* config)
    : m_pIvsandpwAssesser(nullptr)
    , m_pAvadAssesser(nullptr)
    , m_pAsdAndsjdAssesser(nullptr)
    , m_pAadAssesser(nullptr)
    , m_pLaadAssesser(nullptr)
    , m_pMultiLineAssesser(nullptr)
    , m_pMultiLineAortaAssesser(nullptr)
    , m_pPLAXPosClsInferer(nullptr)
    , m_pPLAXAorticQCDetect(nullptr)
{
    std::string plaxClsInferPath;
    if (config->getSpecifiedNode("PLAX_POS_ENGINE_PATH", plaxClsInferPath) && GeneralUtils::fileExists(plaxClsInferPath)) {
        m_pPLAXPosClsInferer = new PLAXPoseClsInferer(plaxClsInferPath);
        QtLogger::instance().logMessage(QString::fromStdString("[I] PLAX Pose Model Loaded"));
    }
    else {
        QtLogger::instance().logMessage(QString::fromStdString("[E] Failed to load PLAX Pose model path from config"));
    }
    // PLAX_Aortic_QC_ENGINE_PATH
    std::string plaxAorticQCInferPath;
    if (config->getSpecifiedNode("PLAX_Aortic_QC_ENGINE_PATH", plaxAorticQCInferPath) && GeneralUtils::fileExists(plaxAorticQCInferPath)) {
        m_pPLAXAorticQCDetect = new PLAXAorticDetect(plaxAorticQCInferPath);
        QtLogger::instance().logMessage(QString::fromStdString("[I] PLAX Aortic Quality Control Model Loaded"));
    }
    else {
        QtLogger::instance().logMessage(QString::fromStdString("[E] Failed to load PLAX Aortic Quality Control model path from config"));
    }
    // 从配置文件中读取 AVAD 模型路径
    std::string avadEnginePath;
    if (config->getSpecifiedNode("AVAD_ENGINE_PATH", avadEnginePath) && GeneralUtils::fileExists(avadEnginePath)) {
        m_pAvadAssesser = new AVADAssess(avadEnginePath);
        QtLogger::instance().logMessage(QString::fromStdString("[I] PLAX AoD Model Loaded"));
    }
    else {
        QtLogger::instance().logMessage(QString::fromStdString("[E] Failed to load AVAD model path from config"));
    }

    // 从配置文件中读取 Aorta 模型路径
    std::string strOutpaintEnginePath, strMultiLineAortaPath;
    if (config->getSpecifiedNode("OUTPAINT_ENGINE_PATH", strOutpaintEnginePath) &&
        config->getSpecifiedNode("MULTILINE_AORTA_ENGINE_PATH", strMultiLineAortaPath)) {

        // 检查 Aorta 模型路径是否存在
        if (GeneralUtils::fileExists(strOutpaintEnginePath) && GeneralUtils::fileExists(strMultiLineAortaPath)) {
            m_pMultiLineAortaAssesser = new MultiLineAssess_Aorta(strOutpaintEnginePath, strMultiLineAortaPath);
            QtLogger::instance().logMessage(QString::fromStdString("[I] PLAX Aorta Model Loaded"));
        }
        else {
            QtLogger::instance().logMessage(QString::fromStdString("[E] Aorta model paths do not exist: ") +
                QString::fromStdString(strOutpaintEnginePath) + ", " +
                QString::fromStdString(strMultiLineAortaPath));
        }
    }
    else {
        QtLogger::instance().logMessage(QString::fromStdString("[E] Failed to load Aorta model paths from config"));
    }

    // place holder 路径（不需要进行检查）
    std::string multiLinePath = "D:/Resources/20240221/param_assess_models/structure_params_line_detectpai_0719.engine";
    std::string detachTwoLinePath = "D:/Resources/20240221/param_assess_models/structure_params_line_detect_0827_2value.engine";
    std::string detachFourLinePath = "D:/Resources/20240221/param_assess_models/structure_params_line_detect_0827_4value.engine";

    // 从配置文件中读取 MultiLine Mask 模型路径
    std::string multiLineMaskPath;
    if (config->getSpecifiedNode("MULTILINE_LV_PATH", multiLineMaskPath)) {
        // 检查 MultiLine Mask 模型路径是否存在
        if (GeneralUtils::fileExists(multiLineMaskPath)) {
            m_pMultiLineAssesser = new MultiLineAssess(multiLinePath, detachTwoLinePath, detachFourLinePath, multiLineMaskPath);
            QtLogger::instance().logMessage(QString::fromStdString("[I] PLAX LV Model Loaded"));
        }
        else {
            QtLogger::instance().logMessage(QString::fromStdString("[E] MultiLine LV model path does not exist: ") +
                QString::fromStdString(multiLineMaskPath));
        }
    }
    else {
        QtLogger::instance().logMessage(QString::fromStdString("[E] Failed to load MultiLine LV model path from config"));
    }
}

PLAXParamsAssess::~PLAXParamsAssess()
{
	delete m_pIvsandpwAssesser;
	delete m_pAvadAssesser;
	delete m_pAsdAndsjdAssesser;
	delete m_pAadAssesser;
	delete m_pLaadAssesser;
    delete m_pMultiLineAssesser;
    delete m_pMultiLineAortaAssesser;
}

int PLAXParamsAssess::getStrucParamsRst(std::vector<cv::Mat>& video, std::vector<int>& keyframeIdx, QMap<QString, QVector<float>>& values, QMap<QString, QImage>& resultPics)
{
    //int index = keyframeIdx.back();
    //if (video.empty())
    //    return 0;

    //if (video.size() <= index)
    //{
    //    m_inferFrame = video.back();
    //}
    //else
    //{
    //    m_inferFrame = video[index];
    //}

    std::vector<cv::Mat> inputVideoClips = video;
    if (inputVideoClips.empty())
        return 0;

    //plax测值输入获取函数
    //getSampledVector(video, inputVideoClips, 5);
    //getKeyframes(keyframeIdx, video, inputVideoClips, 3);
    //parseKeyframes(keyframeIdx, video, inputVideoClips, 5);

    // for test
    //cv::imshow("test", inputVideoClips[0]);
    //cv::waitKey(0);
    //cv::imshow("test1", video[keyframeIdx[0]]);
    //cv::waitKey(0);
    //cv::imshow("test2", m_inferFrame);
    //cv::waitKey(0);
    // for test

    std::map<std::string, std::vector<float>> tempValueMap, AortaValueMap;
    std::map<std::string, cv::Mat> tempImageMap, AortaImageMap;

    //m_multiStructInferer->doInference(inputVideoClips, tempValueMap, tempImageMap);

    //if (inputVideoClips.size() == 0)
    //    std::cout << "size=0" << std::endl;

    // ======== 20250429更新：添加对于ED帧是否能测主动脉或LV的判断 ===========
    std::map<std::string, int> classResults;
    if(m_pPLAXPosClsInferer)
        m_pPLAXPosClsInferer->doInference(*inputVideoClips.begin(), classResults);
    int nCurrPLAXPose = classResults["class"];
    if (nCurrPLAXPose == 1)
    {
        nCurrPLAXPose = m_signal;
        m_signal = 1 - m_signal;
    }

    if (nCurrPLAXPose == 0) // Aortic
    {
        QtLogger::instance().logMessage("[I] PLAX Aortic Position");
        m_pMultiLineAortaAssesser->AortaMultiLineMaskAssessment(inputVideoClips, tempValueMap, tempImageMap);
        aorticQC(inputVideoClips[0], tempValueMap, tempImageMap);
    }
    else if (nCurrPLAXPose == 1)  // Both
    {
        QtLogger::instance().logMessage("[I] PLAX Both Position");
        m_pMultiLineAssesser->multiLineMaskAssessment(inputVideoClips, tempValueMap, tempImageMap);
        m_pMultiLineAortaAssesser->AortaMultiLineMaskAssessment(inputVideoClips, tempValueMap, tempImageMap);
        aorticQC(inputVideoClips[0], tempValueMap, tempImageMap);
    }
    else if (nCurrPLAXPose == 2)  // LV
    {
        QtLogger::instance().logMessage("[I] PLAX LV Position");
        m_pMultiLineAssesser->multiLineMaskAssessment(inputVideoClips, tempValueMap, tempImageMap);
    }

    adjustAADandAoD(nCurrPLAXPose, tempValueMap, tempImageMap);
    //m_pMultiLineAssesser->detachLineAssessment(inputVideoClips, tempValueMap, tempImageMap);
    //m_pMultiLineAssesser->multiLineMaskAssessment(inputVideoClips, tempValueMap, tempImageMap);
    //m_pMultiLineAortaAssesser->AortaMultiLineMaskAssessment(inputVideoClips, tempValueMap, tempImageMap);
    //m_pAvadAssesser->avadAssessment(m_inferFrame, tempValueMap, tempImageMap);

    //try
    //{


    //    //m_pIvsandpwAssesser->ivsAndPwAssessment(m_inferFrame, tempValueMap, tempImageMap);


    //    //m_pAsdAndsjdAssesser->asdAndsjdAssessment(m_inferFrame, tempValueMap, tempImageMap);

    //    //m_pAadAssesser->aadAssessment(m_inferFrame, tempValueMap, tempImageMap);

    //    //m_pLaadAssesser->laadAssessment(m_inferFrame, tempValueMap, tempImageMap);
    //}
    //catch (const std::exception& e) {

    //}
    //std::vector<std::string> vClassNames1 = { "ASD", "SJD", "AAD" };
    //std::vector<std::string> vClassNames2 = { "IVSTd", "LVPWTd", "LVDd", "LAD" };
    //// 定义变量存储筛选结果
    //std::vector<cv::Mat> selectedImages;

    //// 定义变量存储筛选结果
    //cv::Mat imageFromClass1, imageFromClass2, concatMultiImage;

    // 遍历 tempImageMap 筛选图像
    //for (const auto& pair : tempImageMap) {
    //    const std::string& key = pair.first;

    //    // 检查 key 是否属于 vClassNames1
    //    if (imageFromClass1.empty() &&
    //        std::find(vClassNames1.begin(), vClassNames1.end(), key) != vClassNames1.end()) {
    //        imageFromClass1 = pair.second;
    //    }

    //    // 检查 key 是否属于 vClassNames2
    //    if (imageFromClass2.empty() &&
    //        std::find(vClassNames2.begin(), vClassNames2.end(), key) != vClassNames2.end()) {
    //        imageFromClass2 = pair.second;
    //    }

    //    // 如果两种类别都找到，提前退出
    //    if (!imageFromClass1.empty() && !imageFromClass2.empty()) {
    //        selectedImages.push_back(imageFromClass1);
    //        selectedImages.push_back(imageFromClass2);
    //        concatMultiImage=concatMultiImages(selectedImages);
    //        //cv::imshow("11", concatMultiImage);
    //        //cv::waitKey(0);
    //        tempImageMap.emplace("LVDd", concatMultiImage);
    //        selectedImages.clear();

    //        break;
    //    }
    //}

    for (auto& pair : tempValueMap)
    {
        QVector<float> vTempValue(pair.second.begin(), pair.second.end());
        values.insert(QString::fromStdString(pair.first), vTempValue);
    }

    for (auto& pair : tempImageMap)
    {

        //cv::imshow("11",pair.second);
        //cv::waitKey(0);
        resultPics.insert(QString::fromStdString(pair.first), GeneralUtils::matToQImage(pair.second));
    }

	return 0;
}

int PLAXParamsAssess::getSampledVector(const std::vector<cv::Mat>& inputVector, std::vector<cv::Mat>& outputVector, size_t length)
{
    if (inputVector.empty()) {
        return 0;
    }

    size_t inputSize = inputVector.size();

    if (length <= inputSize) 
    {
        double interval = static_cast<double>(inputSize - 1) / (length - 1);
        for (size_t i = 0; i < length; ++i) 
        {
            size_t index = static_cast<size_t>(std::round(i * interval));
            outputVector.push_back(inputVector[index]);
        }
    }
    else 
    {
        if (inputSize > 1)
        {
            double interval = static_cast<double>(inputSize - 1) / (inputSize - 1);
            for (size_t i = 0; i < inputSize; ++i)
            {
                size_t index = static_cast<size_t>(std::round(i * interval));
                outputVector.push_back(inputVector[index]);
            }

            while (outputVector.size() < length)
            {
                if (outputVector.size() % 2 == 0)
                {
                    outputVector.push_back(inputVector.front());
                }
                else
                {
                    outputVector.push_back(inputVector.back());
                }
            }
        }
        else
        {
            while (outputVector.size() < length)
            {
                outputVector.push_back(inputVector.back());
            }
        }
    }

    return 1;
}



int PLAXParamsAssess::getKeyframes(std::vector<int>& keyframeIdx, std::vector<cv::Mat>& inputVideo, std::vector<cv::Mat>& outputVideo, int length)
{

    if (inputVideo.empty()) 
        return 0;

    if (keyframeIdx.empty()) 
    {
        while (outputVideo.size() < length + 2 ) 
        {
            outputVideo.push_back(inputVideo.back());
        }
    }

    size_t videoSize = inputVideo.size();

    // 如果就一个ed索引值，则人为添加第二个ed(x)
    if (keyframeIdx.size() == 1) 
    {
        int keyframeIndex = keyframeIdx.back();
        for (size_t i = keyframeIndex; i < (keyframeIndex + length + 2); ++i)
        {
            if (i <= videoSize)
            {
                outputVideo.push_back(inputVideo[i]);
            }
            else
            {
                outputVideo.insert(outputVideo.begin(), inputVideo[keyframeIndex]);
            }
        }
    }
    else 
    {
        // 以两个ed的中间帧为es
        int first = keyframeIdx[0];
        int second = keyframeIdx[1];
        int middle = first + (second - first) / 2;

        // 生成中间帧索引
        std::vector<int> spacedIndices;
        int count = middle - first - 1; // 2个关键帧之间的实际帧数量

        if (count <= 0)
        {
            while (spacedIndices.size() < length)
            {
                spacedIndices.push_back(first);
            }
        }
        else
        {
            // 等间隔取length或者count数量的点
            for (int i = 1; i <= std::min(length, count); ++i)
            {
                spacedIndices.push_back(first + i * ((middle - first) / (std::min(length, count) + 1)));
            }
            // 保证足额length数量
            while (spacedIndices.size() < 3)
            {
                spacedIndices.push_back(spacedIndices.back());
            }
        }

        outputVideo.push_back(inputVideo[first]);

        for (int idx : spacedIndices)
        {
            outputVideo.push_back(inputVideo[idx]);
        }

        outputVideo.push_back(inputVideo[middle]);
    }
    

    return 1;
}


int PLAXParamsAssess::parseKeyframes(std::vector<int>& keyframeIdx, std::vector<cv::Mat>& inputVideo, std::vector<cv::Mat>& outputVideo, int length)
{
    if (inputVideo.empty()) {
        return 0;
    }

    size_t videoSize = inputVideo.size();
    std::vector<std::pair<int, int>> selectedIdx;

    if (keyframeIdx.size() == 1)  // 只有一个es/ed索引的情况
    {
        if (abs(keyframeIdx.back()) > videoSize) 
        {
            outputVideo.insert(outputVideo.end(), inputVideo.end() - length, inputVideo.end());
            return 0;
        }
    }
    else   // 多个索引值的情况
    {
        std::sort(keyframeIdx.begin(), keyframeIdx.end(), [](int a, int b) {
            return std::abs(a) < std::abs(b);
            });
        for (size_t i = 0; i < keyframeIdx.size() - 1; i++)
        {
            std::pair<int, int> tempIdx;

            if ((abs(keyframeIdx[i]) > videoSize - 1) || (abs(keyframeIdx[i+1]) > videoSize - 1))
                continue;

            if (keyframeIdx[i] > 0 && keyframeIdx[i + 1] < 0)
            {
                tempIdx.first = keyframeIdx[i];
                tempIdx.second = keyframeIdx[i + 1];
                selectedIdx.push_back(tempIdx);
            }
            else
                continue;
        }
    }

    if (!selectedIdx.empty())
    {
        std::pair<int, int> indexPair = selectedIdx.back();

        int edIndex = indexPair.first;
        int esIndex = abs(indexPair.second);
        if (esIndex - edIndex + 1 >= length)  // 两个索引（包含）之间存在length长度帧
        {
            double interval = static_cast<double>(esIndex - edIndex) / (length - 1);
            for (size_t i = 0; i < length; ++i)
            {
                size_t index = static_cast<size_t>(std::round(edIndex + i * interval));
                outputVideo.push_back(inputVideo[index]);
            }
        }
        else   // 小于length
        {
            for (int i = edIndex; i <= esIndex; i++)
            {
                outputVideo.push_back(inputVideo[i]);
            }

            while (outputVideo.size() < length)
            {
                outputVideo.insert(outputVideo.begin(), inputVideo[edIndex]);
            }
        }

    }
    else  // 没有符合条件的ed和es以及keyframeIdx只有单个索引值的情况
    {
        int keyframeIndex = keyframeIdx.back();
        if (keyframeIndex >= 0)  // ed
        {
            for (size_t i = keyframeIndex; i < keyframeIndex + length; ++i)
            {
                if (i <= videoSize - 1)
                {
                    outputVideo.push_back(inputVideo[i]);
                }
                else
                {
                    outputVideo.insert(outputVideo.begin(), inputVideo[keyframeIndex]);
                }
            }
        }
        else   // es
        {
            if (abs(keyframeIndex) + 1 >= length)  // 索引应该是从0开始 
            {
                for (size_t i = abs(keyframeIndex) - length + 1; i < abs(keyframeIndex); ++i)
                {
                    outputVideo.push_back(inputVideo[i]);
                }
            }
            else
            {
                for (size_t i = 0; i < length; ++i)
                {
                    if (i <= abs(keyframeIndex))
                    {
                        outputVideo.push_back(inputVideo[i]);
                    }
                    else
                    {
                        outputVideo.insert(outputVideo.begin(), inputVideo[0]);
                    }
                }
            }

        }
    }

    return 1;

}


//cv::Mat PLAXParamsAssess::concatMultiImages(std::vector<cv::Mat>& vecImages)
//{
//    // 检查所有图像是否有相同的尺寸
//    cv::Size imageSize = vecImages[0].size();
//    for (auto& img : vecImages) {
//        if (img.size() != imageSize) {
//            cv::resize(img, img, imageSize);
//        }
//    }
//
//    // 创建一个大图像，大小为2x2图片拼接
//    int combinedWidth = imageSize.width * 2;
//    int combinedHeight = imageSize.height ;
////    cv::Mat combinedImage(combinedHeight, combinedWidth, vecImages[0].type());
////
////    // 将四张图片复制到大图像中
////    vecImages[0].copyTo(combinedImage(cv::Rect(0, 0, imageSize.width, imageSize.height)));
////    vecImages[1].copyTo(combinedImage(cv::Rect(imageSize.width, 0, imageSize.width, imageSize.height)));
////
////    // 调整拼接后的图像大小以适应目标尺寸
////    cv::Mat resizedCombinedImage;
////    cv::resize(combinedImage, resizedCombinedImage, cv::Size(combinedWidth/2 , combinedHeight ));
////
////    return resizedCombinedImage;
////}
//
//
//cv::Mat PLAXParamsAssess::concatMultiImages(std::vector<cv::Mat>& vecImages)
//{
//    // 检查所有图像是否有相同的尺寸
//    cv::Size imageSize = vecImages[0].size();
//    //for (auto& img : vecImages) {
//    //    if (img.size() != imageSize) {
//    //        cv::resize(img, img, imageSize);
//    //    }
//    //}
//
//    // 创建一个大图像，大小为2x2图片拼接
//    int combinedWidth = imageSize.width * 2;
//    int combinedHeight = imageSize.height;
//    cv::Mat combinedImage(combinedHeight, combinedWidth, vecImages[0].type());
//
//    // 将四张图片复制到大图像中
//    vecImages[0].copyTo(combinedImage(cv::Rect(0, 0, imageSize.width, imageSize.height)));
//    vecImages[1].copyTo(combinedImage(cv::Rect(imageSize.width, 0, imageSize.width, imageSize.height)));
//
//    // 调整拼接后的图像大小以适应目标尺寸
//    cv::Mat resizedCombinedImage;
//    cv::resize(combinedImage, resizedCombinedImage, cv::Size(combinedWidth / 2, combinedHeight/2));
//
//    return resizedCombinedImage;
//}
cv::Mat PLAXParamsAssess::concatMultiImages(std::vector<cv::Mat>& vecImages) {
    if (vecImages.empty() || vecImages.size() < 2) {
        throw std::runtime_error("vecImages should contain at least 2 images.");
    }

    // 检查所有图像是否有相同的尺寸
    cv::Size imageSize = vecImages[0].size();
    for (auto& img : vecImages) {
        if (img.size() != imageSize) {
            cv::resize(img, img, imageSize);
        }
    }

    // 创建一个大图像，大小为2x1图片拼接
    int combinedWidth = imageSize.width * 2;
    int combinedHeight = imageSize.height;
    cv::Mat combinedImage(combinedHeight, combinedWidth, vecImages[0].type());
    if (combinedImage.empty()) {
        throw std::runtime_error("Failed to initialize combinedImage.");
    }

    // 将两张图片复制到大图像中
    vecImages[0].copyTo(combinedImage(cv::Rect(0, 0, imageSize.width, imageSize.height)));
    vecImages[1].copyTo(combinedImage(cv::Rect(imageSize.width, 0, imageSize.width, imageSize.height)));

    // 调整拼接后的图像大小以适应目标尺寸
    cv::Mat resizedCombinedImage;
    if (combinedWidth / 2 <= 0 || combinedHeight / 2 <= 0) {
        throw std::runtime_error("Invalid resize dimensions.");
    }
    cv::resize(combinedImage, resizedCombinedImage, cv::Size(combinedWidth / 2, combinedHeight / 2));

    return resizedCombinedImage;
}

// 20250516更新: 如果中立位，AAD就是AoD；如果是主动脉位，则没有AoD。
int PLAXParamsAssess::adjustAADandAoD(int nPLAXPosCls, std::map<std::string, std::vector<float>>& mapValues, std::map<std::string, cv::Mat>& mapPremiums)
{
    if (nPLAXPosCls == 0)  // Aortic
    {
        mapValues.erase("AoD");
        mapPremiums.erase("AoD");
    }

    else if (nPLAXPosCls == 1)  // Both
    {
        mapValues.erase("AAD");
        mapPremiums.erase("AAD");
    }

    return 1;
}

int PLAXParamsAssess::aorticQC(cv::Mat& src, std::map<std::string, std::vector<float>>& mapValues, std::map<std::string, cv::Mat>& mapPremiums)
{
    std::vector<Object> vecAorticObj;
    if (m_pPLAXAorticQCDetect)
        m_pPLAXAorticQCDetect->doInference(src);
    if (vecAorticObj.empty())
        return 0;

    int nCurrLabel = vecAorticObj[0].label;
    if (vecAorticObj.size() > 1)
    {
        float fMaxConf = 0.0f;
        int nMaxIdx = 0;
        int nCounter = 0;
        for (auto& object : vecAorticObj)
        {
            if (object.conf > fMaxConf)
            {
                fMaxConf = object.conf;
                nMaxIdx = nCounter;
            }
            nCounter++;
        }

        nCurrLabel = vecAorticObj[nMaxIdx].label;
    }

    if (nCurrLabel == 0)
    {
        QtLogger::instance().logMessage(QString("[I] Sinus Quality Bad."));
        std::vector<std::string> aorticRemoveKeys{ "ASD", "SJD" };
        for (auto& key : aorticRemoveKeys)
        {
            mapValues.erase(key);
            mapPremiums.erase(key);
        }
    }
    return 1;
}
