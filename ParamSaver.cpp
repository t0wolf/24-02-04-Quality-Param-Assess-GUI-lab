#include "ParamSaver.h"


ParamSaver::ParamSaver(QString currStudyID)
    : m_rootDir("D:/Data/Shared-Data/System-Prediction")
    , m_filename("")
    , m_currStudyID("")
{
    initParamSaver(currStudyID);
}

ParamSaver::~ParamSaver()
{
}

void ParamSaver::initParamSaver(const QString& stinstID)
{
    if (stinstID.isEmpty())
        return;

    if (stinstID == m_currStudyID)
    {
        return;
    }
    m_currStudyID = stinstID;
    m_filename = QString("%1/%2.xml").arg(m_rootDir, stinstID);

    QDir dir(m_rootDir);
    if (!dir.exists()) {
        if (!dir.mkpath(".")) {
            qDebug() << "Failed to create directory path.";
            return;
        }
    }

    xmlFileInitialize(m_currStudyID);
}

void ParamSaver::addData(const ParamData& data)
{
    QFile file(m_filename);
    if (!file.open(QIODevice::ReadWrite | QIODevice::Text)) {
        qDebug() << "Failed to open file:" << m_filename;
        return;
    }

    QDomDocument doc("PatientData");
    QString errorStr;
    int errorLine;
    int errorColumn;

    if (!doc.setContent(&file, true, &errorStr, &errorLine, &errorColumn)) {
        qDebug() << "Failed to parse XML:" << errorStr << "at line" << errorLine << "column" << errorColumn;
        file.close();
        return;
    }

    file.close();

    QDomElement root = doc.documentElement();
    QString tagName = root.tagName();
    if (root.tagName() != "PatientData") {
        root = doc.createElement("PatientData");
        doc.appendChild(root);
    }

    // 获取根元素
    //QDomElement root = doc.documentElement();

    // 创建新元素
    //QDomElement measurement = doc.createElement("Measurement");

    //QDomElement parameterNode = doc.createElement(data.parameter);
    //parameterNode.appendChild(doc.createTextNode(data.value));

    QDomElement parameterElement = doc.createElement(data.parameter);
    QDomText parameterText = doc.createTextNode(data.value);
    parameterElement.appendChild(parameterText);

    //QDomElement valueElement = doc.createElement("Value");
    //QDomText valueText = doc.createTextNode(data.value);
    //valueElement.appendChild(valueText);

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
        QDomElement measurement = doc.createElement("Measurement");
        measurement.appendChild(parameterElement);
        //measurement.appendChild(valueElement);
        root.appendChild(measurement);
    }

    // 将更新后的XML写回文件
    file.open(QIODevice::WriteOnly);
    QTextStream outStream(&file);
    outStream.setCodec("UTF-8");
    doc.save(outStream, 4); // 缩进4格
    file.close();
}

//void ParamSaver::addData(const ParamData& data)
//{
//    QFile file(m_filename);
//    if (!file.open(QIODevice::ReadWrite | QIODevice::Text)) {
//        qDebug() << "Failed to open file for reading and writing.";
//        return;
//    }
//
//    // 读取现有XML内容
//    QDomDocument doc;
//    if (!doc.setContent(&file)) {
//        qDebug() << "Failed to parse the file into a DOM tree.";
//        file.close();
//        return;
//    }
//
//    // 获取根元素
//    QDomElement root = doc.documentElement();
//
//    // 创建新元素
//    QDomElement newElement = doc.createElement("Measurement");
//
//    // 创建并添加参数项元素
//    QDomElement parameterElement = doc.createElement("Parameter");
//    QDomText parameterText = doc.createTextNode(data.parameter);
//    parameterElement.appendChild(parameterText);
//
//    // 创建并添加测量值元素
//    QDomElement valueElement = doc.createElement("Value");
//    QDomText valueText = doc.createTextNode(data.value);
//    valueElement.appendChild(valueText);
//
//    // 将参数项和测量值添加到新元素中
//    newElement.appendChild(parameterElement);
//    newElement.appendChild(valueElement);
//
//    // 将新元素追加到根元素
//    root.appendChild(newElement);
//
//    // 将更新后的XML写回文件
//    file.resize(0); // 清空文件内容
//    QTextStream outStream(&file);
//    doc.save(outStream, 4); // 缩进4格
//    file.close();
//}

int ParamSaver::xmlFileInitialize(QString & patientName)
{
    QFile file(m_filename);
    if (file.exists()) {
        return 0;
    }

    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qDebug() << "Failed to create file.";
        return 0;
    }

    // 创建一个新文档
    QDomDocument doc;

    // 创建并添加根元素
    QDomElement root = doc.createElement("PatientData");

    // 创建并添加病人姓名元素
    QDomElement nameElement = doc.createElement("PatientName");
    QDomText nameText = doc.createTextNode(patientName);
    nameElement.appendChild(nameText);

    root.appendChild(nameElement);
    doc.appendChild(root);

    // 将XML写入文件
    QTextStream outStream(&file);
    doc.save(outStream, 4); // 缩进4格
    file.close();
    return 1;
}

int ParamSaver::paramHandling(ParamData& paramData)
{
    return 0;
}
