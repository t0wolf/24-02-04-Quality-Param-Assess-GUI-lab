#include "quality_display_widget.h"
#include "ui_quality_display_widget.h"

QualityDisplayWidget::QualityDisplayWidget(QWidget *parent, ProgressSuperThread *progressThread, ConfigParse* configFile)
    : QWidget(parent)
    , ui(new Ui::QualityDisplayWidget)
    , m_progressThread(progressThread)
{
    ui->setupUi(this);

    std::string qualityRulesFilePath = "";
    configFile->getSpecifiedNode("QUALITY_RULES_JSON_PATH", qualityRulesFilePath);
    parseJSONFile(qualityRulesFilePath);
    
    initialize();

    connect(m_progressThread, &ProgressSuperThread::viewNameImageAvailable, this, &QualityDisplayWidget::setCurrentViewNameImage);

    for (auto& viewName : m_viewNameMapping)
    {
        ClickableLabel* label = getViewLabel(viewName);
        connect(label->m_qualityDetailWidget, &QualityDetailWidget::deleteAvailable, this, &QualityDisplayWidget::setViewUncomplete);
        // connect(this, &QualityDisplayWidget::exitThreadAvailable, label, &ClickableLabel::setPlayThreadExit);
        // connect(this, &QualityDisplayWidget::labelInitalizeAvailable, label, &ClickableLabel::setInitialize);
    }

    qInfo() << "QualityDisplayWidget";
    // connect(this, &QualityDisplayWidget::labelInitalizeAvailable, this, &QualityDisplayWidget::setLabelInitialize);
}

QualityDisplayWidget::~QualityDisplayWidget()
{
    delete ui;
}

int QualityDisplayWidget::labelReInit()
{
    initializeBestScores();

    for (auto& viewName : m_viewNameMapping)
    {
        ClickableLabel* qLabel = getViewLabel(viewName);
        if (qLabel)
            emit qLabel->m_qualityDetailWidget->deleteAvailable(viewName);
    }

    m_currentActiveLabel = nullptr;

    return 1;
}

int QualityDisplayWidget::initialize()
{
    initializeBestScores();
    setApperance(ui->a2cLabel, m_viewNameMapping[0]);
    ui->a2cLabel->setViewName("A2C");
    ui->a2cLabel->m_qualityDetailWidget->setQualityRules(m_viewRulesTotal["A2C"].GetObj());

    setApperance(ui->a3cLabel, m_viewNameMapping[1]);
    ui->a3cLabel->setViewName("A3C");
    ui->a3cLabel->m_qualityDetailWidget->setQualityRules(m_viewRulesTotal["A3C"].GetObj());

    setApperance(ui->a4cLabel, m_viewNameMapping[2]);
    ui->a4cLabel->setViewName("A4C");
    ui->a4cLabel->m_qualityDetailWidget->setQualityRules(m_viewRulesTotal["A4C"].GetObj());

    setApperance(ui->a5cLabel, m_viewNameMapping[3]);
    ui->a5cLabel->setViewName("A5C");
    ui->a5cLabel->m_qualityDetailWidget->setQualityRules(m_viewRulesTotal["A5C"].GetObj());

    setApperance(ui->plaxLabel, m_viewNameMapping[4]);
    ui->plaxLabel->setViewName("PLAX");
    ui->plaxLabel->m_qualityDetailWidget->setQualityRules(m_viewRulesTotal["PLAX"].GetObj());

    setApperance(ui->psaxgvLabel, m_viewNameMapping[6]);
    ui->psaxgvLabel->setViewName("PSAXGV");
    ui->psaxgvLabel->m_qualityDetailWidget->setQualityRules(m_viewRulesTotal["PSAXGV"].GetObj());

    setApperance(ui->psaxmvLabel, m_viewNameMapping[7]);
    ui->psaxmvLabel->setViewName("PSAXMV");
    ui->psaxmvLabel->m_qualityDetailWidget->setQualityRules(m_viewRulesTotal["PSAXMV"].GetObj());

    setApperance(ui->psaxpmLabel, m_viewNameMapping[8]);
    ui->psaxpmLabel->setViewName("PSAXPM");
    ui->psaxpmLabel->m_qualityDetailWidget->setQualityRules(m_viewRulesTotal["PSAXPM"].GetObj());

    setApperance(ui->psaxaLabel, m_viewNameMapping[5]);
    ui->psaxaLabel->setViewName("PSAXA");
    ui->psaxaLabel->m_qualityDetailWidget->setQualityRules(m_viewRulesTotal["PSAXA"].GetObj());

    return 1;
}

int QualityDisplayWidget::setApperance(QLabel* label, const QString& dispText)
{
    label->clear();
    QLabel* textLabel = nullptr;
    textLabel = label->findChild<QLabel*>();

    if (textLabel)
    {
        delete textLabel;
        textLabel = nullptr;
    }

    QPalette pal = palette();
    pal.setColor(QPalette::Window, Qt::gray);
    label->setPalette(pal);
    label->setAutoFillBackground(true);
    //<p align=\"center\">
    QString htmlText = QString("<html><head/><body>"
                               "<span style='color: #808080; font-weight: bold; font-size: %2;'>%1</span>"
                               "</p></body></html>").arg(dispText, FONT_SIZE);
    label->setText(htmlText);
    label->setAlignment(Qt::AlignLeft | Qt::AlignTop);

    // 设置QLabel允许显示富文本
    label->setTextFormat(Qt::RichText);

    // 通过设置样式表进一步自定义QLabel的外观（如果需要）
    label->setStyleSheet("QLabel { background-color: transparent; border: 2px solid white; border-radius: 4px;}");
    return 1;
}

int QualityDisplayWidget::setApperance(QLabel* label, QPixmap& qImage, const QString& dispText, const QString color)
{
    label->clear(); // 这将清除label的内容，包括文本和图像
    // label->setPixmap(QPixmap());
    // label->setPixmap(qImage.scaled(label->size()));
    label->setAutoFillBackground(true);
    label->setPixmap(qImage.scaled(label->width() - 30, label->height() - 30));

    QLabel *textLabel = nullptr;
    textLabel = label->findChild<QLabel*>();

    if (textLabel)
    {
        delete textLabel;
        textLabel = nullptr;
    }

    textLabel = new QLabel(label); // 将文本QLabel作为图像QLabel的子控件
    textLabel->raise();
    textLabel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    textLabel->setStyleSheet(
        QString("QLabel {"
            "   border: none;"
            "   color: white;"
            "   font-weight: bold;"
            //"   font-family: Arial;"
            "   font-size: %1;"
            "}").arg(FONT_SIZE)
        );
    textLabel->setText(QString("<h1 style='color: %2;'>%1</h1>").arg(dispText, color));

    textLabel->resize(label->width(), textLabel->height());
    textLabel->setAlignment(Qt::AlignLeft | Qt::AlignTop); // 文本顶部居中对齐
    textLabel->show();

    return 1;
}

int QualityDisplayWidget::setApperance(QLabel* label, const QString& dispText, const QString color)
{
    // QString htmlText = QString("<html><head/><body><p align=\"center\" style='margin-top: 0;'>"
    //                            "<span style='color: %1; font-size: 20px; font-family: Arial;'>%2</span>"
    //                            "</p></body></html>").arg(color, dispText);
    // label->setText(htmlText);
    // label->setAlignment(Qt::AlignTop | Qt::AlignHCenter); // 确保文本在QLabel的顶部中央对齐

    // label->setTextFormat(Qt::RichText);

    // label->setStyleSheet("QLabel { background-color: transparent; border: 2px solid white; border-radius: 4px;}");
    // if (m_viewCompleteMap[dispText] == false)

    QLabel *textLabel = nullptr;
    textLabel = label->findChild<QLabel*>();

    if (textLabel)
    {
        delete textLabel;
        textLabel = nullptr;
    }

    textLabel = new QLabel(label); // 将文本QLabel作为图像QLabel的子控件
    textLabel->raise();
    textLabel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    textLabel->setStyleSheet(
        QString("QLabel {"
            "   border: none;" // 去除边框
            "   color: white;" // 文字颜色
            "   font-weight: bold;" // 字体加粗
            //"   font-family: Arial;"
            "   font-size: %1;" // 字体大小
            "}").arg(FONT_SIZE)
        );
    textLabel->setText(QString("<h1 style='color: %2;'>%1</h1>").arg(dispText, color));

    textLabel->resize(label->width(), textLabel->height());
    textLabel->setAlignment(Qt::AlignLeft | Qt::AlignTop); // 文本顶部居中对齐
    textLabel->show();
    return 1;
    

    //return 0;
}

ClickableLabel* QualityDisplayWidget::getViewLabel(const QString& viewName)
{
    if (viewName == "A2C")
        return ui->a2cLabel;
    else if (viewName == "A3C")
        return ui->a3cLabel;
    else if (viewName == "A4C")
        return ui->a4cLabel;
    else if (viewName == "A5C")
        return ui->a5cLabel;
    else if (viewName == "PLAX")
        return ui->plaxLabel;
    else if (viewName == "PSAXA")
        return ui->psaxaLabel;
    else if (viewName == "PSAXGV")
        return ui->psaxgvLabel;
    else if (viewName == "PSAXPM")
        return ui->psaxpmLabel;
    else if (viewName == "PSAXMV")
        return ui->psaxmvLabel;
    else
        return new ClickableLabel();
}

void QualityDisplayWidget::setActiveLabel(ClickableLabel* label)
{
    disconnect(this, &QualityDisplayWidget::exitThreadAvailable, nullptr, nullptr);
    disconnect(this, &QualityDisplayWidget::labelInitalizeAvailable, nullptr, nullptr);
    disconnect(this, &QualityDisplayWidget::qualityScoresAvailable, nullptr, nullptr);

    m_currentActiveLabel = label;
    if (m_currentActiveLabel)
    {
        connect(this, &QualityDisplayWidget::exitThreadAvailable, m_currentActiveLabel, &ClickableLabel::setPlayThreadExit);
        connect(this, &QualityDisplayWidget::labelInitalizeAvailable, m_currentActiveLabel, &ClickableLabel::setInitialize);
        connect(this, &QualityDisplayWidget::qualityScoresAvailable, m_currentActiveLabel, &ClickableLabel::setQualityScores);
    }
}

void QualityDisplayWidget::setCurrentViewNameVideo(const QString viewName, QVariant qVar)
{
    if (viewName.isEmpty())
        return;

    if (m_progressThread->getViewProgress(viewName) == false)
    {
        QLabel* qLabel = getViewLabel(viewName);

        if (qLabel == nullptr)
            return;

        setApperance(qLabel, viewName, "#00FF00");
        auto videoClips = qVar.value<QVector<cv::Mat>>();
        if (videoClips.empty())
            return;

        if (viewName == "A2C")
        {
            // m_a2cVideoClip = videoClips;
            ui->psaxmvLabel->m_videoClipBuffer.clear();
            ui->a2cLabel->m_videoClipBuffer = videoClips;
        }
        else if (viewName == "A3C")
        {
            // m_a3cVideoClip = videoClips;
            ui->psaxmvLabel->m_videoClipBuffer.clear();
            ui->a3cLabel->m_videoClipBuffer = videoClips;
        }
        else if (viewName == "A4C")
        {
            // m_a4cVideoClip = videoClips;
            ui->psaxmvLabel->m_videoClipBuffer.clear();
            ui->a4cLabel->m_videoClipBuffer = videoClips;
        }
        else if (viewName == "A5C")
        {
            // m_a5cVideoClip = videoClips;
            ui->psaxmvLabel->m_videoClipBuffer.clear();
            ui->a5cLabel->m_videoClipBuffer = videoClips;
        }
        else if (viewName == "PLAX")
        {
            // m_plaxVideoClip = videoClips;
            ui->psaxmvLabel->m_videoClipBuffer.clear();
            ui->plaxLabel->m_videoClipBuffer = videoClips;
        }
        else if (viewName == "PSAXA")
        {
            // m_psaxaVideoClip = videoClips;
            ui->psaxmvLabel->m_videoClipBuffer.clear();
            ui->psaxaLabel->m_videoClipBuffer = videoClips;
        }
        else if (viewName == "PSAXGV")
        {
            // m_psaxgvVideoClip = videoClips;
            ui->psaxmvLabel->m_videoClipBuffer.clear();
            ui->psaxgvLabel->m_videoClipBuffer = videoClips;
        }
        else if (viewName == "PSAXPM")
        {
            ui->psaxpmLabel->m_videoClipBuffer.clear();
            ui->psaxpmLabel->m_videoClipBuffer = videoClips;
        }
        else if (viewName == "PSAXMV")
        {
            // m_psaxmvVideoClip = videoClips;
            ui->psaxmvLabel->m_videoClipBuffer.clear();
            ui->psaxmvLabel->m_videoClipBuffer = videoClips;
        }

        // m_viewCompleteMap[viewName] = true;
        //m_progressThread->setViewQualityComplete(viewName);
    }

}

void QualityDisplayWidget::setCurrentViewQualityControlled(const QString viewName, QVariant qVideoClips, QVariant qVResult)
{
    QVector<cv::Mat> videoClips = qVideoClips.value<QVector<cv::Mat>>();
    QVector<float> vQualityScores = qVResult.value<QVector<float>>();
    QPair<bool, bool> qualityScoreResult = judgeQualityScores(viewName, vQualityScores);

    bool isQualityControl, isBestScore;
    isQualityControl = qualityScoreResult.first;
    isBestScore = qualityScoreResult.second;

    QString color = "";
    if (isQualityControl)
    {
        color = GREEN_COLOR;  // green
        m_progressThread->setViewQualityComplete(viewName);
    }
    else
        color = RED_COLOR;  // red

    if (isBestScore || !m_progressThread->getViewProgress(viewName))
    {
        setViewNameVideo(viewName, videoClips, color);
        setQualityScores(viewName, qVResult);
    }
}

int QualityDisplayWidget::parseJSONFile(std::string jsonPath)
{
    std::ifstream ifs(jsonPath);
    if (!ifs.is_open())
    {
        return 0;
    }

    std::stringstream ss;
    ss << ifs.rdbuf();
    ifs.close();

    if (m_viewRulesTotal.Parse(ss.str().c_str()).HasParseError())
    {
        return 0;
    }

    return 1;
}

void QualityDisplayWidget::setViewUncomplete(const QString viewName)
{
    // m_viewCompleteMap[viewName] = false;
    m_progressThread->setViewQualityUncomplete(viewName);
    ClickableLabel* label = getViewLabel(viewName);
    if (label)
    {
        setActiveLabel(label);
        emit exitThreadAvailable();
        emit labelInitalizeAvailable();
    }
}

void QualityDisplayWidget::setCurrentViewName(const QString viewName)
{
    if (viewName.isEmpty())
        return;

    QLabel* qLabel = getViewLabel(viewName);

    if (qLabel == nullptr)
        return;

    setApperance(qLabel, viewName, "#FFD700");
}

void QualityDisplayWidget::setCurrentViewNameImage(const QString viewName, QImage qImage)
{
    // if (m_viewCompleteMap[viewName] == false)
    if (viewName.isEmpty())
        return;

    if (m_progressThread->getViewProgress(viewName) == false)
    {
        QLabel* qLabel = getViewLabel(viewName);

        if (qLabel == nullptr)
            return;

        QPixmap tempPixmap = QPixmap::fromImage(qImage);
        setApperance(qLabel, tempPixmap, viewName, "#FFD700");
    }

}

void QualityDisplayWidget::setLabelInitialize(const QString viewName)
{
    if (viewName.isEmpty())
        return;

    if (m_progressThread->getViewQualityStatus(viewName))
        return;
    QLabel* qLabel = getViewLabel(viewName);

    if (qLabel == nullptr)
        return;

    setApperance(qLabel, viewName);
}

void QualityDisplayWidget::setQualityScores(QString viewName, QVariant qVResult)
{
    if (viewName.isEmpty())
        return;

    ClickableLabel* label = getViewLabel(viewName);
    setActiveLabel(label);
    emit qualityScoresAvailable(qVResult);
}

int QualityDisplayWidget::setViewNameVideo(const QString& viewName, QVector<cv::Mat>& vVideoCLips, QString& color)
{
    if (viewName.isEmpty())
        return 0;

    QLabel* qLabel = getViewLabel(viewName);
    if (qLabel == nullptr)
        return 0;

    setApperance(qLabel, viewName, color);
    if (vVideoCLips.empty())
        return 0;

    if (viewName == "A2C")
    {
        ui->psaxmvLabel->m_videoClipBuffer.clear();
        ui->a2cLabel->m_videoClipBuffer = vVideoCLips;
    }
    else if (viewName == "A3C")
    {
        ui->psaxmvLabel->m_videoClipBuffer.clear();
        ui->a3cLabel->m_videoClipBuffer = vVideoCLips;
    }
    else if (viewName == "A4C")
    {
        ui->psaxmvLabel->m_videoClipBuffer.clear();
        ui->a4cLabel->m_videoClipBuffer = vVideoCLips;
    }
    else if (viewName == "A5C")
    {
        ui->psaxmvLabel->m_videoClipBuffer.clear();
        ui->a5cLabel->m_videoClipBuffer = vVideoCLips;
    }
    else if (viewName == "PLAX")
    {
        ui->psaxmvLabel->m_videoClipBuffer.clear();
        ui->plaxLabel->m_videoClipBuffer = vVideoCLips;
    }
    else if (viewName == "PSAXA")
    {
        ui->psaxmvLabel->m_videoClipBuffer.clear();
        ui->psaxaLabel->m_videoClipBuffer = vVideoCLips;
    }
    else if (viewName == "PSAXGV")
    {
        ui->psaxmvLabel->m_videoClipBuffer.clear();
        ui->psaxgvLabel->m_videoClipBuffer = vVideoCLips;
    }
    else if (viewName == "PSAXPM")
    {
        ui->psaxpmLabel->m_videoClipBuffer.clear();
        ui->psaxpmLabel->m_videoClipBuffer = vVideoCLips;
    }
    else if (viewName == "PSAXMV")
    {
        ui->psaxmvLabel->m_videoClipBuffer.clear();
        ui->psaxmvLabel->m_videoClipBuffer = vVideoCLips;
    }

    return 1;
}

QPair<bool, bool> QualityDisplayWidget::judgeQualityScores(const QString& viewName, QVector<float>& vResults)
{
    QPair<bool, bool> currControlBestResult = QPair<bool, bool>(false, false);
    float fTotalScore = 0.0f;
    for (auto& fScore : vResults)
        fTotalScore += fScore;

    float fCurrViewBestScore = m_viewBestQualityScores[viewName];
    if (fTotalScore > fCurrViewBestScore)
    {
        currControlBestResult.second = true;
        m_viewBestQualityScores[viewName] = fTotalScore;
    }

    if (fTotalScore >= m_viewQualityThresh[viewName])
        currControlBestResult.first = true;

    return currControlBestResult;
}
