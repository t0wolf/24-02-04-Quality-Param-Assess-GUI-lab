#include "ParamAssessHTMLShower.h"

ParamAssessHTMLShower::ParamAssessHTMLShower(QObject *parent, QString strDataRootPath)
	: QObject(parent)
	, m_strRootPath(strDataRootPath)
    , m_strHTMLFilePath(strDataRootPath + "/html")
    , m_strImageSaveRootPath(strDataRootPath + "/saved-param-premiums")
    , m_strCurrInstID("")
    , m_strPrevInstID("")
    , m_strCurrPatientName("Unknown")
    , m_strPrevPatientName("Unknown")
    , m_lastSpecSaveTime(QDateTime::currentDateTime().addSecs(-30))
    , m_lastStructSaveTime(QDateTime::currentDateTime().addSecs(-30))
{}

ParamAssessHTMLShower::~ParamAssessHTMLShower()
{}

int ParamAssessHTMLShower::updateParamValues(QString strViewName, QMap<QString, QString>&mapParamValues, QMap<QString, QImage>&mapParamPremiums)
{
    if (mapParamPremiums.isEmpty())
        return 0;

    QString strHtmlContent = loadExistHtmlFile().toUtf8();

    QString measurementEntries = "";

    for (auto& strParamEvent : mapParamValues.keys())
    {
        measurementEntries += QString(
            "<div class='measurement-item'><strong>%1:</strong> %2</div>"
        ).arg(strParamEvent).arg(mapParamValues[strParamEvent]);
    }

    // 临时增加参数显示的逻辑，如果有PLAX的结构参数，则优先先显示IVS+LVID+LVPW的预览图
    bool bHasStructParam = false;
    for (auto& strParamEvent : mapParamPremiums.keys())
    {
        if (strParamEvent == "IVSTd")
        {
            bHasStructParam = true;
            break;
        }
    }

    QImage paramDisplayPremiums = QImage();
    
    if (bHasStructParam)
        paramDisplayPremiums = mapParamPremiums["IVSTd"];
    else
        paramDisplayPremiums = mapParamPremiums.begin().value();
    QString strPremiumSavePath = savePremiumImageToDisk(strViewName, paramDisplayPremiums);
    // =========================================================================

    /*QString strPremiumSavePath = savePremiumImageToDisk(strViewName, mapParamPremiums.begin().value());*/

    QString sliceEntry = QString(
        "<div class='container'>"
        "<div class='image-container'><img src='%1' alt='AI测值预览图' /></div>"
        "<div class='measurements-container'>%2</div>"
        "</div>"
    ).arg(strPremiumSavePath).arg(measurementEntries);

    // Find the end of the body tag and insert the new measurement entry before it
    int nInsertPos = strHtmlContent.lastIndexOf("</body>");
    if (nInsertPos != -1)
    {
        strHtmlContent.insert(nInsertPos, sliceEntry);
    }

    saveHtmlFile(strHtmlContent);
    
    return 1;
}

void ParamAssessHTMLShower::slotReceiveParamValuesPremiums(QString viewName, QVariant paramValues, QVariant paramPremium)
{
    QMap<QString, QString> qmParamValues = paramValues.value<QMap<QString, QString>>();
    QMap<QString, QImage> qmParamPremiums = paramPremium.value<QMap<QString, QImage>>();

    updateParamValues(viewName, qmParamValues, qmParamPremiums);
}

void ParamAssessHTMLShower::slotReceiveStudyInstID(const QString& strStudyInstID)
{
    QMutexLocker locker(&m_mutex);
    m_strPrevInstID = m_strCurrInstID;
    m_strCurrInstID = strStudyInstID;
}

void ParamAssessHTMLShower::slotReceivePatientName(const QString& strPatientName)
{
    QMutexLocker locker(&m_mutex);
    m_strPrevPatientName = m_strCurrPatientName;
    m_strCurrPatientName = strPatientName;
}

QString ParamAssessHTMLShower::loadHtmlTemplate()
{
    return QString(
        "<html>"
        "<head><title>Measurement Results</title></head>"
        "<body>"
        "<h1>Ultrasound Measurement Results</h1>"
        "<img src='{{IMAGE_PATH}}' alt='Measurement Image' /><br>"
        "<table border='1'>"
        "<tr><th>Parameter</th><th>Value</th></tr>"
        "{{MEASUREMENTS}}"
        "</table>"
        "</body>"
        "</html>"
    );
}

QString ParamAssessHTMLShower::createBaseHtml()
{
    return QString(
        "<html>"
        "<head>"
        "<meta charset=\"UTF-8\">"
        "<title>Measurement Results</title>"
        "<style>"
        "body { font-family: Arial; margin: 0; padding: 0; background-color: #f0f4f8; color: #333; }"
        "header { background-color: #007bff; color: white; padding: 10px 0; text-align: center; margin-bottom: 20px; }"
        "h1 { margin: 0; }"
        ".container { display: flex; align-items: flex-start; justify-content: center; margin: 20px 0; padding: 15px; background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); max-width: 800px; margin-left: auto; margin-right: auto; }"
        ".image-container { flex: 1; margin-right: 20px; text-align: center; }"
        ".image-container img { max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #ddd; }"
        ".measurements-container { flex: 1; padding: 10px; }"
        ".measurement-item { background-color: #f8f9fa; padding: 10px; margin-bottom: 10px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); }"
        ".measurement-item strong { display: block; font-size: 1.1em; color: #007bff; }"
        "</style>"
        "</head>"
        "<body>"
        "<header><h1>AI超声参数测量</h1></header>"
        "</body>"
        "</html>"
    );
}

QString ParamAssessHTMLShower::loadExistHtmlFile()
{
    QString strHtmlContent;
    QFile htmlFile(m_strHTMLFilePath + "/" + m_strCurrInstID + ".html");
    if (htmlFile.exists() && htmlFile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QTextStream in(&htmlFile);
        in.setCodec("UTF-8");
        strHtmlContent = in.readAll();
        htmlFile.close();
    }

    else
    {
        strHtmlContent = createBaseHtml();
    }

    return strHtmlContent;
}

int ParamAssessHTMLShower::saveHtmlFile(QString& strHtmlContent)
{
    QFile htmlFile(m_strHTMLFilePath + "/" + m_strCurrInstID + ".html");
    if (htmlFile.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        QTextStream out(&htmlFile);
        out.setCodec("UTF-8");
        out << strHtmlContent;
        htmlFile.close();
    }

    return 1;
}

QString ParamAssessHTMLShower::savePremiumImageToDisk(QString& strEventName, QImage& img)
{
    cv::Mat cvImage = GeneralUtils::qImage2cvMat(img);
    QString strSaveImgPath = saveSpecInferenceImage(strEventName, cvImage, true);

    return strSaveImgPath;
}


QString ParamAssessHTMLShower::saveSpecInferenceImage(QString& eventName, cv::Mat& img, bool isPremium)
{
    if (eventName.indexOf("JG") != -1)
        eventName = "TDIMVIVS";
    else if (eventName.indexOf("CB") != -1)
        eventName = "TDIMVLW";
    else if (eventName.indexOf("VTI") != -1)
        return 0;

    if (img.empty())
        return 0;

    QString fileName = "";
    if (isPremium)
        fileName = GeneralUtils::generateDate() + "_premium_" + ".jpg";
    else
        fileName = GeneralUtils::generateDate() + ".jpg";

    QString strImageURL = "/saved-param-premiums/" + m_strCurrPatientName + "/" + eventName + "/" + fileName;
    QString folderPath = m_strImageSaveRootPath + "/" + m_strCurrPatientName + "/" + eventName;
    QDir().mkpath(folderPath);

    QString filePath = folderPath + "/" + fileName;
    cv::imwrite(filePath.toStdString(), img);
    return strImageURL;
}

int ParamAssessHTMLShower::saveStructInferenceImage(QString viewName,
    std::vector<cv::Mat>& vInferVideoClip,
    std::vector<int>& vKeyframeIndexes,
    QMap<QString, QImage>& qmPremiums)
{
    QDateTime currentDataTime = QDateTime::currentDateTime();
    QString currentDate = currentDataTime.toString("yyyyMMdd"); // 当天日期，格式：年月日
    QString currentTime = currentDataTime.toString("HHmmss"); // 当前时间，格式：时分秒

    QString currDateTime = GeneralUtils::generateDate();
    QString videoSaveFolderName = m_structImageSaveRootPath + "/origin_videos/" + viewName + "/" + currDateTime;
    //QString premiumSaveFolderName = m_structImageSaveRootPath + "/premiums/" + viewName + "/" + currDateTime;
    QString premiumSaveFolderName = m_structImageSaveRootPath + "/premiums/" + viewName + "/" + currentDate + "/" + m_strCurrInstID + "/" + currentTime;
    QDir().mkpath(videoSaveFolderName);
    QDir().mkpath(premiumSaveFolderName);

    int counter = 0;
    for (auto& frame : vInferVideoClip)
    {
        QString currFrameSavePath = "";
        if (std::find(vKeyframeIndexes.begin(), vKeyframeIndexes.end(), counter) != vKeyframeIndexes.end())
            currFrameSavePath = videoSaveFolderName + "/" + QString("image_%1_keyframe.jpg").arg(counter, 5, 10, QChar('0'));
        else
            currFrameSavePath = videoSaveFolderName + "/" + QString("image_%1.jpg").arg(counter, 5, 10, QChar('0'));
        cv::imwrite(currFrameSavePath.toStdString(), frame);
        ++counter;
    }

    for (auto& eventName : qmPremiums.keys())
    {
        QString currFrameSavePath = premiumSaveFolderName + "/" + QString("premium_%1.jpg").arg(eventName);
        cv::Mat currPremium = GeneralUtils::qImage2cvMat(qmPremiums[eventName]);
        cv::imwrite(currFrameSavePath.toStdString(), currPremium);
    }

    return 1;
}