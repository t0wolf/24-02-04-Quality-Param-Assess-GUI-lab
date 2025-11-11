#include "ParamTransmitter.h"

ParamTransmitter::ParamTransmitter(ConfigParse* configParser, QObject* parent)
    : QObject(parent)
    , m_currStudyInstID("")
    , m_currPatientName("")
    , m_currPatientID("")
    , m_workStationIP("")
    , m_workStationPort("")
{
    m_netManager = new QNetworkAccessManager(this);
    std::string workStationIP, workStationPort;
    configParser->getSpecifiedNode("SERVER_IP_ADDRESS", workStationIP);
    configParser->getSpecifiedNode("SERVER_PORT", workStationPort);

    m_workStationIP = QString::fromStdString(workStationIP);
    m_workStationPort = QString::fromStdString(workStationPort);
}

ParamTransmitter::ParamTransmitter(ConfigParse* configParser, PatientManagerWindow* patientManageWindow, QObject* parent)
    : QObject(parent)
    , m_currStudyInstID("")
    , m_currPatientName("")
    , m_currPatientID("")
    , m_workStationIP("")
    , m_workStationPort("")
{
    m_netManager = new QNetworkAccessManager(this);
    std::string workStationIP, workStationPort;
    configParser->getSpecifiedNode("SERVER_IP_ADDRESS", workStationIP);
    configParser->getSpecifiedNode("SERVER_PORT", workStationPort);

    m_workStationIP = QString::fromStdString(workStationIP);
    m_workStationPort = QString::fromStdString(workStationPort);
    m_patientManageWindow = patientManageWindow;
}

ParamTransmitter::~ParamTransmitter()
{

}

void ParamTransmitter::initParamTransmitter(const QString & stinstID)
{
    if (stinstID.isEmpty())
        return;

    if (stinstID == m_currStudyInstID)
    {
        return;
    }
    else
    {
        m_currStudyInstID = stinstID;
        clearParamJsonArray();
    }

    initDocument();
}

void ParamTransmitter::initParamTransmitter(const QString& stinstID, const QString& patientName, const QString& patientID)
{
    if (stinstID.isEmpty())
        return;

    if (stinstID == m_currStudyInstID)
    {
        return;
    }

    m_currStudyInstID = stinstID;
    m_currPatientName = patientName;
    m_currPatientID = patientID;

    initDocument();
}

int ParamTransmitter::addDataAndSendToServer(const ParamData& paramData)
{
    if (!sendDataToServer(paramData))
        return 0;

    return 1;
}

int ParamTransmitter::addDataToDoc(const ParamData& data)
{
    QDomElement root = m_currXmlData.documentElement();
    QString tagName = root.tagName();
    if (root.tagName() != "PatientData") {
        root = m_currXmlData.createElement("PatientData");
        m_currXmlData.appendChild(root);
    }
    
    QDomElement parameterElement = m_currXmlData.createElement(data.parameter);
    QDomText parameterText = m_currXmlData.createTextNode(data.value);
    parameterElement.appendChild(parameterText);

    QDomNodeList measurements = root.elementsByTagName("Measurement");
    bool found = false;

    for (int i = 0; i < measurements.count(); ++i)
    {
        QDomElement currentMeasurement = measurements.at(i).toElement();
        QDomNodeList parameters = currentMeasurement.elementsByTagName(data.parameter);

        if (parameters.count() > 0)
        {
            parameters.at(0).firstChild().setNodeValue(data.value);
            found = true;
            break;
        }
    }

    if (!found)
    {
        for (int i = 0; i < measurements.count(); ++i)
        {
            QDomElement currentMeasurement = measurements.at(i).toElement();
            currentMeasurement.appendChild(parameterElement);
            //currentMeasurement.appendChild(valueElement);
            found = true;
            break;
        }
    }

    if (!found)
    {
        QDomElement measurement = m_currXmlData.createElement("Measurement");
        measurement.appendChild(parameterElement);
        //measurement.appendChild(valueElement);
        root.appendChild(measurement);
    }

    return 1;
}

int ParamTransmitter::addDataToJson(const ParamData& data)
{
    bool found = false;
    ParamData processedData = handleFuncParamsNames(data);

    if (m_patientManageWindow != nullptr)
    {
        m_patientManageWindow->updatePatientMeasurement(processedData.parameter, processedData.value.toFloat());
    }

    // 遍历数组，检查是否已存在对应参数
    for (int i = 0; i < m_currJsonArray.size(); ++i) 
    {
        QJsonObject currentObject = m_currJsonArray[i].toObject();

        if (currentObject["name"].toString() == processedData.parameter) 
        {
            currentObject["value"] = processedData.value;
            m_currJsonArray[i] = currentObject;
            found = true;
            break;
        }
    }

    if (!found) 
    {
        // 如果没有找到，创建一个新的对象
        QJsonObject newObject;
        newObject["name"] = processedData.parameter;
        newObject["value"] = processedData.value;
        m_currJsonArray.append(newObject);
    }
    //QJsonArray newJsonArray;
    //QJsonObject newObject;
    //newObject["name"] = processedData.parameter;
    //newObject["value"] = processedData.value;
    //newJsonArray.append(newObject);
    //m_currJsonArray = newJsonArray;

    return 1;
}

int ParamTransmitter::sendDataToServer(const QString& strViewName, const QImage& premiumImage)
{
    //if (m_currPatientID != m_)
    QString strPremiumBase64Code = QString::fromLatin1(imageToBase64(premiumImage));

    // 直接使用 QJsonArray，而不是将其转换为字符串
    QJsonDocument jsonDoc(m_currJsonArray);
    QByteArray aimeterageData = jsonDoc.toJson(QJsonDocument::Compact);

    QString strCurrQualityView, strCurrQualityScore;
    {
        QMutexLocker locker(&m_mutex);
        strCurrQualityView = m_currQualityViewName;
        strCurrQualityScore = m_currQualityScore;
    }

    if (strCurrQualityScore.isEmpty())
        strCurrQualityScore = "60";

    QString currUrlStr = QString("http://%1:%2/updateAIMeterage").arg(m_workStationIP).arg(m_workStationPort);
    QUrl currUrl(currUrlStr);

    QtLogger::instance().logMessage(QString("[I] Param Transmit - %1 %2 %3 %4 %5")
        .arg(m_currStudyInstID)
        .arg(m_currPatientName)
        .arg(m_currPatientID)
        .arg(strViewName)
        .arg(strCurrQualityScore));

    QNetworkRequest request(currUrl);

    // 创建一个多部分表单数据
    QHttpMultiPart* multiPart = new QHttpMultiPart(QHttpMultiPart::FormDataType);

    // 添加其他字段作为表单数据
    QHttpPart studyInstanceIdPart;
    studyInstanceIdPart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"studyinstanceid\""));
    studyInstanceIdPart.setBody(m_currStudyInstID.toUtf8());
    multiPart->append(studyInstanceIdPart);

    QHttpPart patientNamePart;
    patientNamePart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"patientname\""));
    patientNamePart.setBody(m_currPatientName.toUtf8());
    multiPart->append(patientNamePart);

    QHttpPart patientIdPart;
    patientIdPart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"patientid\""));
    patientIdPart.setBody(m_currPatientID.toUtf8());
    multiPart->append(patientIdPart);

    QHttpPart imageTypePart;
    imageTypePart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"imagetype\""));
    imageTypePart.setBody("png");
    multiPart->append(imageTypePart);

    QHttpPart viewPart;
    viewPart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"view\""));
    viewPart.setBody(strViewName.toUtf8());
    multiPart->append(viewPart);

    QHttpPart qualityScorePart;
    qualityScorePart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"qualityscore\""));
    qualityScorePart.setBody(strCurrQualityScore.toUtf8());
    multiPart->append(qualityScorePart);

    // 添加 premium image 作为表单数据
    QHttpPart premiumImagePart;
    premiumImagePart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"premiumimage\""));
    premiumImagePart.setBody(strPremiumBase64Code.toUtf8());
    multiPart->append(premiumImagePart);

    // 添加 aimeterage data 作为表单数据
    QHttpPart aimeteragePart;
    aimeteragePart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"aimeterage\""));
    aimeteragePart.setHeader(QNetworkRequest::ContentTypeHeader, QVariant("application/json"));
    aimeteragePart.setBody(aimeterageData);
    multiPart->append(aimeteragePart);

    QNetworkReply* reply = m_netManager->post(request, multiPart);
    multiPart->setParent(reply); // 确保 multiPart 在 reply 被删除时也删除

    QtLogger::instance().logMessage(QString("[I] Param Values: %1").arg(QString::fromUtf8(aimeterageData)));

    connect(reply, &QNetworkReply::finished, this, &ParamTransmitter::handleNetworkReply);
    clearParamJsonArray();

    return 1;
}

void ParamTransmitter::handleNetworkReply()
{
    QNetworkReply* reply = qobject_cast<QNetworkReply*>(sender());
    if (reply)
    {
        if (reply->error() == QNetworkReply::NoError) 
        {
            qDebug() << "Request succeeded:" << reply->readAll();
        }
        else 
        {
            qDebug() << "Request failed:" << reply->errorString();
        }
        reply->deleteLater();
    }
}

int ParamTransmitter::sendDataToServer(const ParamData& paramData)
{
    //addDataToDoc(paramData);
    clearParamJsonArray();
    addDataToJson(paramData);
    sendDataToServer();

    return 1;
}

int ParamTransmitter::sendDataToServer()
{
    if (m_workStationIP.isEmpty() || m_workStationPort.isEmpty() || m_currStudyInstID.isEmpty())
        return 0;

    QString currXMLDataString = m_currXmlData.toString();
    QString currEncodedData = base64Encode(currXMLDataString);

    QString currUrlStr = QString("http://%1:%2/updateAIMeterage").arg(m_workStationIP).arg(m_workStationPort);
    QUrl currUrl(currUrlStr);
    QUrlQuery currUrlQuery;

    currUrlQuery.addQueryItem("studyInstanceID", m_currStudyInstID);
    currUrlQuery.addQueryItem("aimeterage", currEncodedData);
    currUrl.setQuery(currUrlQuery);

    QNetworkRequest request(currUrl);
    QNetworkReply* reply = m_netManager->get(request);

    connect(reply, &QNetworkReply::finished, this, &ParamTransmitter::handleNetworkReply);
    return 1;
}

void ParamTransmitter::slotHandleQualityScore(QString qualityViewName, QString qualityScore)
{
    QMutexLocker locker(&m_mutex);
    m_currQualityScore = qualityScore;
    m_currQualityViewName = qualityViewName;
}

ParamData ParamTransmitter::handleFuncParamsNames(const ParamData& paramData)
{
    ParamData processedParamData = paramData;

    bool conversionOk;
    double value = paramData.value.toDouble(&conversionOk);
    if (conversionOk)  // 调整保留两位小数，TDI保留3位
    {
        if (paramData.parameter.contains("JG") || paramData.parameter.contains("CB"))
            processedParamData.value = QString::number(value, 'f', 3);
        else
            processedParamData.value = QString::number(value, 'f', 2);
    }
    else
    {
        return processedParamData;
    }


    if (paramData.parameter.contains("EDV") || paramData.parameter.contains("ESV") || paramData.parameter.contains("EF"))
    {
        QString originParamEvent = paramData.parameter.split(" ")[0];
        processedParamData.parameter = originParamEvent;
    }
    return processedParamData;
}

//int ModelsInferenceThread::computeEDivedeA()
//{
//    bool bIsEAParam = false;
//    for (auto it = m_currSignalParamValues.begin(); it != m_currSignalParamValues.end(); ++it)
//    {
//        if (it.key() == QString("E") || it.key() == QString("A"))
//        {
//            bIsEAParam = true;
//            break;
//        }
//    }
//
//    if (!bIsEAParam)
//        return 0;
//    else
//    {
//        if (!m_currSignalParamValues.contains("E") || !m_currSignalParamValues.contains("A"))
//            return 0;
//        if (m_currSignalParamValues["A"] == QString("0.0"))
//            return 0;
//        float fE = m_currSignalParamValues["E"].toFloat();
//        float fA = m_currSignalParamValues["A"].toFloat();
//        float fEDivideA = fE / fA;
//        m_currSignalParamValues.insert("E/A", QString::number(fEDivideA));
//    }
//    return 1;
//}

int ParamTransmitter::clearParamJsonArray()
{
    QJsonArray newJsonArray;
    m_currJsonArray = newJsonArray;
    return 0;
}

void ParamTransmitter::initDocument()
{
    m_currXmlData.clear();
    QDomElement root = m_currXmlData.createElement("PatientData");

    // 创建并添加病人姓名元素
    QDomElement nameElement = m_currXmlData.createElement("PatientName");
    QDomText nameText = m_currXmlData.createTextNode(m_currStudyInstID);
    nameElement.appendChild(nameText);

    root.appendChild(nameElement);
    m_currXmlData.appendChild(root);
}

QByteArray ParamTransmitter::imageToBase64(const QString& imagePath)
{
    QImage image(imagePath);
    if (image.isNull()) {
        qWarning() << "Failed to load image from path:" << imagePath;
        return QByteArray();
    }
    return imageToBase64(image);
}

QByteArray ParamTransmitter::imageToBase64(const QImage& image)
{
    QByteArray byteArray;
    QBuffer buffer(&byteArray);
    image.save(&buffer, "PNG");  // You can specify the format you need
    return byteArray.toBase64();
}
