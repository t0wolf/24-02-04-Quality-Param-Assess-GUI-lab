#include "param_display_widget.h"
#include "ui_param_display_widget.h"

ParamDisplayWidget::ParamDisplayWidget(QWidget *parent,
    ProgressSuperThread *progressThread,
    ParamBrowserWidget* paramBrowserWidget,
    ParamTransmitter* paramTransmitter)
    : QWidget(parent)
    , ui(new Ui::ParamDisplayWidget)
    , m_progressThread(progressThread)
    , m_paramBrowserWidget(paramBrowserWidget)
    , m_paramSaver(new ParamSaver(QString("")))
    , m_paramTransmitter(paramTransmitter)
{
    ui->setupUi(this);
    ui->paramListWidget->setSpacing(0);
    parseJSONFile("D:\\Resources\\20240221\\param_assess_events.json");

    initialize();
    connect(m_progressThread, &ProgressSuperThread::sigParamValuePremiumsAvailable, this, &ParamDisplayWidget::setParamValuesPics);
    connect(m_progressThread, &ProgressSuperThread::sigStructParamValuePremiumsAvailable, this, &ParamDisplayWidget::setStructParamValuesPics);
    connect(this->ui->structureParamListWidget, &QListWidget::itemPressed, this, &ParamDisplayWidget::onItemPressed);

    QScroller::grabGesture(ui->structureParamListWidget->viewport(), QScroller::TouchGesture);
    QScroller::grabGesture(ui->paramListWidget->viewport(), QScroller::TouchGesture);

    qInfo() << "ParamDisplayWidget";
}

ParamDisplayWidget::~ParamDisplayWidget()
{
    delete ui;
}

int ParamDisplayWidget::paramLabelsReInit()
{
    for (int i = 0; i < ui->paramListWidget->count(); i++)
    {
        QListWidgetItem* item = ui->paramListWidget->item(i);
        ParamDisplayItemWidget* widget = qobject_cast<ParamDisplayItemWidget*>(ui->paramListWidget->itemWidget(item));
        if (widget)
        {
            widget->initializeParamEvents();
            m_progressThread->setParamUncomplete(widget->getName());
        }
    }

    for (int i = 0; i < ui->structureParamListWidget->count(); i++)
    {
        QListWidgetItem* item = ui->structureParamListWidget->item(i);
        ParamDisplayItemWidget* widget = qobject_cast<ParamDisplayItemWidget*>(ui->structureParamListWidget->itemWidget(item));
        if (widget)
        {
            widget->initializeParamEvents();
            m_progressThread->setParamUncomplete(widget->getName());
        }
    }
    return 1;
}

void ParamDisplayWidget::saveParamToCsv(const QString& rootPath, const QString& patientName)
{
    if (m_qmTotalParamValues.isEmpty())
        return;

    QString dateFolderName = getCurrentDateFolderName();
    QDir dir(rootPath);
    QString subFolderPath = dir.filePath(dateFolderName);

    // 检查并创建子文件夹
    QDir subDir(subFolderPath);
    if (!subDir.exists()) {
        if (!subDir.mkpath(".")) {
            std::cerr << "Failed to create directory: " << subFolderPath.toStdString() << std::endl;
            return;
        }
    }

    // 创建文件路径
    //QString filePath = subDir.filePath(getCurrentDateTimeFileName());
    QString filePath;
    if (!patientName.isEmpty())
        filePath = QDir(subDir).filePath(patientName + "_" + getCurrentDateFileName() + ".csv");
    else
        filePath = QDir(subDir).filePath(QString("Unknown") + "_" + getCurrentDateFileName() + ".csv");
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        std::cerr << "Couldn't open file for writing: " << filePath.toStdString() << std::endl;
        return;
    }

    QTextStream out(&file);
    out.setCodec("UTF-8");

    // 写入表头
    QStringList headers = m_qmTotalParamValues.keys();
    out << headers.join(",") << "\n";

    // 获取最长列的长度
    int maxRows = 0;
    for (const auto& values : m_qmTotalParamValues) {
        if (values.size() > maxRows) {
            maxRows = values.size();
        }
    }

    // 写入数据
    for (int row = 0; row < maxRows; ++row) {
        QStringList rowValues;
        for (const auto& key : m_qmTotalParamValues.keys()) {
            const QVector<QString>& values = m_qmTotalParamValues[key];
            if (row < values.size()) {
                rowValues << values[row];
            }
            else {
                rowValues << ""; // 如果该列没有对应的值，用空字符串填充
            }
        }
        out << rowValues.join(",") << "\n";
    }

    if (file.error() != QFile::NoError) {
        std::cerr << "Error writing to file: " << file.errorString().toStdString() << std::endl;
    }
    else {
        std::cout << "CSV file saved successfully at: " << filePath.toStdString() << std::endl;
    }

    file.close();
}

int ParamDisplayWidget::initialize()
{
    initParamEvents();
    m_progressThread->setParamList(m_paramEvents);
    for (auto& key : m_structParamEvents.keys())
    {
        QString viewName = key;
        QVector<QString> paramEvents = m_structParamEvents[viewName];

        for (auto& paramEvent : paramEvents)
        {
            QString widgetName = paramEvent;
            QListWidgetItem *item = new QListWidgetItem();
            ui->structureParamListWidget->addItem(item);

            ParamDisplayItemWidget *paramDisplayItemWidget = new ParamDisplayItemWidget();
            paramDisplayItemWidget->setName(widgetName);
            ui->structureParamListWidget->setItemWidget(item, paramDisplayItemWidget);
            item->setSizeHint(paramDisplayItemWidget->sizeHint());

            // connect signals to slots.
            connect(paramDisplayItemWidget->m_paramPreWidget, &ParamPremiumWidget::sigDeleteParam, m_progressThread, &ProgressSuperThread::setParamUncomplete);
            connect(paramDisplayItemWidget->m_paramPreWidget, &ParamPremiumWidget::sigDeleteParam, this, &ParamDisplayWidget::setParamValueDeleted);
        }
    }

    for (auto& key : m_spectrumParamEvents.keys())
    {
        QString viewName = key;
        QVector<QString> paramEvents = m_spectrumParamEvents[viewName];

        for (auto& paramEvent : paramEvents)
        {
            QString widgetName = paramEvent;
            QListWidgetItem *item = new QListWidgetItem();
            ui->paramListWidget->addItem(item);

            ParamDisplayItemWidget *paramDisplayItemWidget = new ParamDisplayItemWidget();
            paramDisplayItemWidget->setName(widgetName);
            ui->paramListWidget->setItemWidget(item, paramDisplayItemWidget);
            item->setSizeHint(paramDisplayItemWidget->sizeHint());

            // connect signals to slots.
            connect(paramDisplayItemWidget->m_paramPreWidget, &ParamPremiumWidget::sigDeleteParam, m_progressThread, &ProgressSuperThread::setParamUncomplete);
            // connect();
        }
    }

    for (auto& key : m_funcParamEvents.keys())
    {
        QString viewName = key;
        QVector<QString> paramEvents = m_funcParamEvents[viewName];

        for (auto& paramEvent : paramEvents)
        {
            QString widgetName = paramEvent;
            QListWidgetItem *item = new QListWidgetItem();
            ui->paramListWidget->addItem(item);

            ParamDisplayItemWidget *paramDisplayItemWidget = new ParamDisplayItemWidget();
            paramDisplayItemWidget->setName(widgetName);
            ui->paramListWidget->setItemWidget(item, paramDisplayItemWidget);
            item->setSizeHint(paramDisplayItemWidget->sizeHint());

            // connect signals to slots.
            connect(paramDisplayItemWidget->m_paramPreWidget, &ParamPremiumWidget::sigDeleteParam, m_progressThread, &ProgressSuperThread::setParamUncomplete);
            // connect();
        }
    }

    return 1;
}

int ParamDisplayWidget::initParamEvents()
{
    if (m_paramEvents.IsObject())
    {
        for (rapidjson::Value::ConstMemberIterator itr = m_paramEvents.MemberBegin(); itr != m_paramEvents.MemberEnd(); ++itr)
        {
            QString key = itr->name.GetString();

            // Structure Parameters
            if (itr->value.HasMember("structure_params") && itr->value["structure_params"].IsArray()) {
                const rapidjson::Value& strcutParamsValue = itr->value["structure_params"];
                QVector<QString> vParams;

                for (rapidjson::SizeType i = 0; i < strcutParamsValue.Size(); i++) {
                    vParams.push_back(strcutParamsValue[i].GetString());
                }
                m_structParamEvents.insert(key, vParams);
            }

            // Spectrum Parameters
            if (itr->value.HasMember("spectrum_params") && itr->value["spectrum_params"].IsArray()) {
                const rapidjson::Value& spectrumParamsValue = itr->value["spectrum_params"];
                QVector<QString> vParams;

                for (rapidjson::SizeType i = 0; i < spectrumParamsValue.Size(); i++) {
                    vParams.push_back(spectrumParamsValue[i].GetString());
                }
                m_spectrumParamEvents.insert(key, vParams);
            }

            // Functional Parameters
            if (itr->value.HasMember("functional_params") && itr->value["functional_params"].IsArray()) {
                const rapidjson::Value& funcParamsValue = itr->value["functional_params"];
                QVector<QString> vParams;

                for (rapidjson::SizeType i = 0; i < funcParamsValue.Size(); i++) {
                    vParams.push_back(funcParamsValue[i].GetString());
                }
                m_funcParamEvents.insert(key, vParams);
            }
        }
    }
    return 1;
}

void ParamDisplayWidget::handleFuncParam(QString& strParamName)
{
    if (strParamName.contains("EDV") || strParamName.contains("ESV") || strParamName.contains("EF"))
    {
        QString strRealFuncParamName = strParamName.split(" ")[0];
        changeFuncParamWidgetName(strRealFuncParamName, strParamName);
    }
}

void ParamDisplayWidget::changeFuncParamWidgetName(QString strCurrParamName, QString strChangedParamName)
{
    ParamDisplayItemWidget* objFuncWidget = nullptr;
    for (int i = 0; i < ui->paramListWidget->count(); i++)
    {
        QListWidgetItem* item = ui->paramListWidget->item(i);
        ParamDisplayItemWidget* widget = qobject_cast<ParamDisplayItemWidget*>(ui->paramListWidget->itemWidget(item));
        if (widget && widget->getName().contains(strCurrParamName))
        {
            objFuncWidget = widget;
            break;
        }
    }

    if (objFuncWidget)
    {
        objFuncWidget->setName(strChangedParamName);
    }
}

//void ParamDisplayWidget::paramBrowserClicked()
//{
//    m_paramBrowserWidget->show();
//}

int ParamDisplayWidget::parseJSONFile(std::string jsonPath)
{
    std::ifstream ifs(jsonPath);
    if (!ifs.is_open())
    {
        return 0;
    }

    std::stringstream ss;
    ss << ifs.rdbuf();
    ifs.close();

    if (m_paramEvents.Parse(ss.str().c_str()).HasParseError())
    {
        return 0;
    }

    return 1;
}

ParamDisplayItemWidget* ParamDisplayWidget::findSpecificParamWidget(QString paramName)
{
    ParamDisplayItemWidget* objectItem = nullptr;
    for (int i = 0; i < ui->paramListWidget->count(); i++)
    {
        QListWidgetItem *item = ui->paramListWidget->item(i);
        ParamDisplayItemWidget* widget = qobject_cast<ParamDisplayItemWidget*>(ui->paramListWidget->itemWidget(item));
        if (widget && widget->getName() == paramName)
        {
            objectItem = widget;
            break;
        }
    }

    for (int i = 0; i < ui->structureParamListWidget->count(); i++)
    {
        QListWidgetItem *item = ui->structureParamListWidget->item(i);
        ParamDisplayItemWidget* widget = qobject_cast<ParamDisplayItemWidget*>(ui->structureParamListWidget->itemWidget(item));
        if (widget && widget->getName() == paramName)
        {
            objectItem = widget;
            break;
        }
    }

    return objectItem;
}

int ParamDisplayWidget::addParamToSaver(ParamData& paramData)
{
    // MV TDI
    if (paramData.parameter == "TDIMVIVS" || paramData.parameter == "TDIMVLW" || paramData.parameter == "MV")
    {
        QStringList splittedValues = paramData.value.split(" ");
        if (splittedValues.size() > 1)
        {
            if ((paramData.parameter == "TDIMVIVS" || paramData.parameter == "TDIMVLW") && splittedValues.size() == 3)
            {
                QString newParamEvent = paramData.parameter + QString("_%1").arg("S");
                //m_paramSaver->addData({ newParamEvent, splittedValues[0] });
                m_paramTransmitter->addDataAndSendToServer({ newParamEvent, splittedValues[0] });

                newParamEvent = paramData.parameter + QString("_%1").arg("E");
                //m_paramSaver->addData({ newParamEvent, splittedValues[1] });
                m_paramTransmitter->addDataAndSendToServer({ newParamEvent, splittedValues[1] });

                newParamEvent = paramData.parameter + QString("_%1").arg("A");
                //m_paramSaver->addData({ newParamEvent, splittedValues[2] });
                m_paramTransmitter->addDataAndSendToServer({ newParamEvent, splittedValues[2] });
            }

            else if (paramData.parameter == "MV" && splittedValues.size() == 2)
            {
                QString newParamEvent = paramData.parameter + QString("_%1").arg("E");
                //m_paramSaver->addData({ newParamEvent, splittedValues[0] });
                m_paramTransmitter->addDataAndSendToServer({ newParamEvent, splittedValues[0] });

                newParamEvent = paramData.parameter + QString("_%1").arg("A");
                //m_paramSaver->addData({ newParamEvent, splittedValues[1] });
                m_paramTransmitter->addDataAndSendToServer({ newParamEvent, splittedValues[1] });
            }
        }

        else
        {
            //m_paramSaver->addData({ paramData.parameter, paramData.value });
            m_paramTransmitter->addDataAndSendToServer({ paramData.parameter, paramData.value });
        }
    }  // if (paramData.parameter == "TDIMVIVS" || paramData.parameter == "TDIMVLW" || paramData.parameter == "MV")

    else
    {
        //m_paramSaver->addData({ paramData.parameter, paramData.value });
        m_paramTransmitter->addDataAndSendToServer({ paramData.parameter, paramData.value });
    }
    return 1;
}

int ParamDisplayWidget::addParamToTransmitter(QMap<QString, QString>& paramEventValue)
{
    for (auto& paramEvent : paramEventValue.keys())
    {
        ParamData paramData({ paramEvent, paramEventValue[paramEvent] });
        m_paramTransmitter->addDataToDoc(paramData);
    }
    m_paramTransmitter->sendDataToServer();
    return 1;
}

int ParamDisplayWidget::addParamToTransmitter(QString& strViewName, QMap<QString, QString>& paramEventValue, QMap<QString, QImage>& paramEventPremium)
{
    if (paramEventPremium.isEmpty() || paramEventValue.isEmpty())
        return 0;

    const QImage currPremium = paramEventPremium.first();
    const QString currParamType = paramEventPremium.firstKey();

    for (auto& paramEvent : paramEventValue.keys())
    {
        ParamData paramData({ paramEvent, paramEventValue[paramEvent] });
        m_paramTransmitter->addDataToJson(paramData);
    }
    m_paramTransmitter->sendDataToServer(strViewName, currPremium);
    return 0;
}

void ParamDisplayWidget::setParamValues(const QString viewName, QVariant qVar)
{
    QMap<QString, float> qmParamValues = qVar.value<QMap<QString, float>>();

    for (auto& paramEvent : qmParamValues.keys())
    {
        ParamDisplayItemWidget* objectWidget = findSpecificParamWidget(paramEvent);
        if (objectWidget != nullptr)
            objectWidget->setParamValue(qmParamValues[paramEvent]);
    }

}

void ParamDisplayWidget::saveParamJsonToFile(const QString& rootPath, const QString& patientName)
{
    QJsonObject jsonObject;

    if (m_qmTotalParamValues.isEmpty())
        return;

    // 遍历QMap，将其转换为QJsonObject
    for (auto it = m_qmTotalParamValues.begin(); it != m_qmTotalParamValues.end(); ++it) {
        QJsonArray jsonArray;
        for (const QString& value : it.value()) {
            jsonArray.append(value);
        }
        jsonObject.insert(it.key(), jsonArray);
    }

    // 创建QJsonDocument并保存到文件
    QJsonDocument jsonDocument(jsonObject);

    QString filePath;
    if (!patientName.isEmpty())
        filePath = QDir(rootPath).filePath(patientName + "_" + getCurrentDateFileName() + ".json");
    else
        filePath = QDir(rootPath).filePath(QString("Unknown") + "_" + getCurrentDateFileName() + ".json");
    QFile file(filePath);

    if (!file.open(QIODevice::WriteOnly)) {
        std::cerr << "Couldn't open file for writing." << std::endl;
        return;
    }

    file.write(jsonDocument.toJson(QJsonDocument::Indented));
    file.close();
}

void ParamDisplayWidget::initParamSaver(QString patientName)
{
    if (m_paramSaver != nullptr)
    {
        m_paramSaver->initParamSaver(patientName);
    }
    else
    {
        m_paramSaver = new ParamSaver(patientName);
    }
}

void ParamDisplayWidget::setParamValuesPics(QString viewName, QVariant qValues, QVariant qPremiums)
{
    QMap<QString, QString> qmParamValues = qValues.value<QMap<QString, QString>>();
    QMap<QString, QImage> qmParamPremiums = qPremiums.value<QMap<QString, QImage>>();
    //addParamToTransmitter(qmParamValues);
    addParamToTransmitter(viewName, qmParamValues, qmParamPremiums);

    for (auto& paramEvent : qmParamValues.keys())
    {
        ParamData paramData({ paramEvent, qmParamValues[paramEvent] });
        //addParamToSaver(paramData);

        if (m_qmTotalParamValues.contains(paramEvent))
        {
            m_qmTotalParamValues[paramEvent].push_back(qmParamValues[paramEvent]);
        }
        else
        {
            m_qmTotalParamValues[paramEvent] = QVector<QString>{ qmParamValues[paramEvent] };
        }

        // 处理改名后的功能参数
        handleFuncParam(paramEvent);

        ParamDisplayItemWidget* objectWidget = findSpecificParamWidget(paramEvent);
        if (objectWidget != nullptr)
        {
            objectWidget->setParamValue(qmParamValues[paramEvent]);
            objectWidget->setParamPics(qmParamPremiums[paramEvent]);

            m_progressThread->setParamComplete(paramEvent);
            //m_paramBrowserWidget->updateImage(qmParamPremiums[paramEvent], paramEvent);
        }

        //if (m_progressThread->getParamProgress(paramEvent) == false)
        //{
        //    ParamDisplayItemWidget* objectWidget = findSpecificParamWidget(paramEvent);
        //    if (objectWidget != nullptr)
        //    {
        //        objectWidget->setParamValue(qmParamValues[paramEvent]);
        //        objectWidget->setParamPics(qmParamPremiums[paramEvent]);

        //        m_progressThread->setParamComplete(paramEvent);
        //        m_paramBrowserWidget->updateImage(qmParamPremiums[paramEvent], paramEvent);
        //    }
        //}
    }
}

void ParamDisplayWidget::setStructParamValuesPics(QVariant qValues, QVariant qPremiums)
{
    QMap<QString, QMap<QString, QString>> mapParamValues = qValues.value<QMap<QString, QMap<QString, QString>>>();
    QMap<QString, QImage> mapParamPremiums = qPremiums.value<QMap<QString, QImage>>();
    //addParamToTransmitter(qmParamValues);

    for (auto& key : mapParamValues.keys())
    {
        QMap<QString, QString> qmParamValues = mapParamValues[key];
        QMap<QString, QImage> qmParamPremiums;
        qmParamPremiums[key] = mapParamPremiums[key];

        QString viewName = QString("PLAX");
        addParamToTransmitter(viewName, qmParamValues, qmParamPremiums);

        for (auto& paramEvent : qmParamValues.keys())
        {
            ParamData paramData({ paramEvent, qmParamValues[paramEvent] });
            //addParamToSaver(paramData);

            if (m_qmTotalParamValues.contains(paramEvent))
            {
                m_qmTotalParamValues[paramEvent].push_back(qmParamValues[paramEvent]);
            }
            else
            {
                m_qmTotalParamValues[paramEvent] = QVector<QString>{ qmParamValues[paramEvent] };
            }

            ParamDisplayItemWidget* objectWidget = findSpecificParamWidget(paramEvent);
            if (objectWidget != nullptr)
            {
                objectWidget->setParamValue(qmParamValues[paramEvent]);
                objectWidget->setParamPics(qmParamPremiums[paramEvent]);

                m_progressThread->setParamComplete(paramEvent);
                m_paramBrowserWidget->updateImage(qmParamPremiums[paramEvent], paramEvent);
            }
        }
    }
}

void ParamDisplayWidget::setParamValueDeleted(QString paramName)
{
    ParamDisplayItemWidget* objectWidget = findSpecificParamWidget(paramName);
    if (objectWidget != nullptr)
        objectWidget->setParamValueDeleted();
}
