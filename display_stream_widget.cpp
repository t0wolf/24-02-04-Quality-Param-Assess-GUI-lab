#include "display_stream_widget.h"
#include "ui_display_stream_widget.h"
#include <QDebug>
#include <QGraphicsView>

DisplayStreamWidget::DisplayStreamWidget(QualityDisplayWidget* qualityDisplayWidget, 
    ParamDisplayWidget* paramDisplayWidget,
    ProgressSuperThread *progressThread, 
    ConfigParse* config,
    QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::DisplayStreamWidget)
    , m_infoExtractThread(new InfoExtractThread(this))
    , m_qualityDisplayWidget(qualityDisplayWidget)
    , m_paramDisplayWidget(paramDisplayWidget)
    , m_textItem(new QGraphicsTextItem())
    , m_paramAssessHtmlShower(new ParamAssessHTMLShower(parent, QDir::currentPath()))
{
    m_roiDataBuffer = new DataBuffer();
    m_roiScaleThread = new ROIScaleDetThread(this, m_roiDataBuffer);
    m_modelsInferThread = new ModelsInferenceThread(this, progressThread, m_roiDataBuffer, config);
    ui->setupUi(this);

    ui->displayView->setScene(&m_scene);

    m_textItem->setDefaultTextColor(Qt::white);
    m_textItem->setFont(QFont("Arial", 20));
    m_textItem->setPos(10, 10);
    m_scene.addItem(m_textItem);

    // 创建视频拉流线程
    m_videoThread = new VideoStreamThread(this, m_roiScaleThread, m_modelsInferThread, m_infoExtractThread, config);

    // 连接图像帧信号到QLabel的槽函数
    connect(m_videoThread, &VideoStreamThread::frameAvailable, this, &DisplayStreamWidget::displayFrame);
    connect(m_modelsInferThread, &ModelsInferenceThread::sigDebugText, this, &DisplayStreamWidget::displayDebugText);

    //connect(m_infoExtractThread, &InfoExtractThread::sigScaleInfoAvailable, m_modelsInferThread, &ModelsInferenceThread::setScaleInfo);
    connect(m_infoExtractThread, &InfoExtractThread::sigScaleModeInfoAvailable, m_modelsInferThread, &ModelsInferenceThread::setScaleModeInfo);
    connect(m_modelsInferThread, &ModelsInferenceThread::sigParamsAvailable, m_modelsInferThread->m_progressThread, &ProgressSuperThread::setCurrentParamValuePics);
    connect(m_modelsInferThread, &ModelsInferenceThread::sigStructParamsAvailable, m_modelsInferThread->m_progressThread, &ProgressSuperThread::setStructParamValuePics);
    connect(m_modelsInferThread, &ModelsInferenceThread::sigParamsAvailable, m_paramAssessHtmlShower, &ParamAssessHTMLShower::slotReceiveParamValuesPremiums);

    connect(m_roiScaleThread, &ROIScaleDetThread::sigROIScaleAvailable, m_infoExtractThread, &InfoExtractThread::setROIScaleInfo);
    //connect(m_roiScaleThread, &ROIScaleDetThread::sigROIScaleAvailable, m_modelsInferThread, &ModelsInferenceThread::setROIScaleInfo, Qt::DirectConnection);

    // connect(m_modelsInferThread, &ModelsInferenceThread::structureParamsAvailable, m_modelsInferThread->m_progressThread, &ProgressSuperThread::setCurrentParam);
    // connect(m_modelsInferThread, &ModelsInferenceThread::viewNameVideoAvailable, m_modelsInferThread->m_progressThread, &ProgressSuperThread::setCurrentViewNameVideo);
    connect(m_modelsInferThread, &ModelsInferenceThread::viewNameProcessed, m_modelsInferThread->m_progressThread, &ProgressSuperThread::setCurrentViewName);
    connect(m_modelsInferThread, &ModelsInferenceThread::viewNameImageAvailable, m_modelsInferThread->m_progressThread, &ProgressSuperThread::setCurrentViewNameImage);

    //connect(m_modelsInferThread, &ModelsInferenceThread::viewNameVideoAvailable, m_qualityDisplayWidget, &QualityDisplayWidget::setCurrentViewNameVideo);
    connect(m_modelsInferThread, &ModelsInferenceThread::sigViewNameImageVideoAvailable, m_qualityDisplayWidget, &QualityDisplayWidget::setCurrentViewQualityControlled);
    //connect(m_modelsInferThread, &ModelsInferenceThread::qualityScoresAvailable, m_qualityDisplayWidget, &QualityDisplayWidget::setQualityScores);

    connect(m_modelsInferThread, &ModelsInferenceThread::sigQualityInput, m_modelsInferThread->m_qualityControlThread, &QualityControlThread::setQualityInput);
    connect(m_modelsInferThread->m_qualityControlThread, &QualityControlThread::sigVideoResult, m_modelsInferThread, &ModelsInferenceThread::setQualityControlScores);
    connect(m_modelsInferThread, &ModelsInferenceThread::sigParamInput, m_modelsInferThread->m_paramAssessThread, &ParamAssessThread::setParamAssessInput);
    connect(m_modelsInferThread->m_paramAssessThread, &ParamAssessThread::sigParamsResult, m_modelsInferThread, &ModelsInferenceThread::setParamsAssessValues);

    //connect(m_infoExtractThread, &InfoExtractThread::sigPatientInfo, m_modelsInferThread->m_paramAssessThread, &ParamAssessThread::setPatientName);
    connect(m_modelsInferThread, &ModelsInferenceThread::sigScaleInfo, m_modelsInferThread->m_paramAssessThread, &ParamAssessThread::setScaleInfo);

    deployAllInferThreads();
}

DisplayStreamWidget::~DisplayStreamWidget()
{
    delete ui;
    m_videoThread->exitThread();
    m_roiScaleThread->exitThread();
    m_modelsInferThread->exitThread();
    m_infoExtractThread->exitThread();
    m_modelsInferThread->m_progressThread->exitThread();

    delete m_videoThread;
    delete m_roiScaleThread;
    delete m_modelsInferThread;
    delete m_infoExtractThread;
}

void DisplayStreamWidget::adaptToView()
{
    QRectF rectItem = m_scene.itemsBoundingRect();
    QRectF rectView = ui->displayView->rect();
    qreal ratioView = rectView.height() / rectView.width();
    qreal ratioItem = rectItem.height() / rectItem.width();
    if (ratioView > ratioItem)
    {
        rectItem.moveTop(rectItem.width()*ratioView - rectItem.height());
        rectItem.setHeight(rectItem.width()*ratioView);

        rectItem.setWidth(rectItem.width() * 1.0);
        rectItem.setHeight(rectItem.height() * 1.0);
    }
    else
    {
        rectItem.moveLeft(rectItem.height()/ratioView - rectItem.width());
        rectItem.setWidth(rectItem.height()/ratioView);

        rectItem.setWidth(rectItem.width() * 1.0);
        rectItem.setHeight(rectItem.height() * 1.0);
    }
    ui->displayView->fitInView(rectItem, Qt::KeepAspectRatio);
}

int DisplayStreamWidget::inputFramesToThreads(const QImage& qImage)
{
    cv::Mat cvFrame = GeneralUtils::qImage2cvMat(qImage);
    m_roiScaleThread->inputVideoFrame(cvFrame);
    m_modelsInferThread->inputVideoFrame(cvFrame);
    m_infoExtractThread->inputVideoFrame(cvFrame);

    return 1;
}

int DisplayStreamWidget::deployAllInferThreads()
{
    m_modelsInferThread->m_progressThread->start();
    m_videoThread->start();
    m_roiScaleThread->start();
    m_modelsInferThread->start();
    m_infoExtractThread->start();
    ui->beginButton->setText(QString::fromLocal8Bit("结束检查"));

    return 1;
}

int DisplayStreamWidget::disableAllInferThreads()
{
    m_videoThread->exitThread();
    m_roiScaleThread->exitThread();
    m_modelsInferThread->exitThread();
    m_infoExtractThread->exitThread();
    m_modelsInferThread->m_progressThread->exitThread();
    ui->beginButton->setText(QString::fromLocal8Bit("开始检查"));
    return 1;
}

void DisplayStreamWidget::displayFrame(const QImage qImage)
{
    //inputFramesToThreads(qImage);

    QPixmap tempPixmap = QPixmap::fromImage(qImage);
    m_scene.clear();
    m_scene.addPixmap(tempPixmap);

    // Re-create the text item to ensure it's valid
    m_textItem = new QGraphicsTextItem();
    m_textItem->setDefaultTextColor(Qt::white);
    m_textItem->setFont(QFont("Arial", 20));
    m_textItem->setPos(10, 10);
    //m_scene.addItem(m_textItem);

    m_scene.addItem(m_textItem);  // Re-add the text item after clearing the scene
    if (!m_debugText.isEmpty())
        m_textItem->setPlainText(m_debugText);
    adaptToView();
    // ui->displayLabel->setPixmap(tempPixmap);
}

void DisplayStreamWidget::displayDebugText(QString text)
{
    // 单行日志
    //m_debugText = text;
    // 多行日志
    if (text == m_debugTextbuffer)
    {
        m_debugText = m_debugTexts.join("\n");
    }
    else
    {
        m_debugTexts.append(text);

        if (m_debugTexts.size() > 8)
            m_debugTexts.removeFirst();

        m_debugTextbuffer = text;
        m_debugText = m_debugTexts.join("\n");
    }
}

void DisplayStreamWidget::on_beginButton_clicked()
{
    // 启动视频拉流线程
    if (m_videoThread->getVideoCapStatus())
    {
        disableAllInferThreads();
    }
    else
    {
        deployAllInferThreads();
    }
}

void DisplayStreamWidget::on_readVideoButton_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, "选择视频文件", "",
        "视频文件 (*.mp4 *.avi *.mkv *.flv *.mov);;所有文件 (*)");

    if (fileName.isEmpty())
        return;

    // 检查文件路径是否存在
    if (!QFile::exists(fileName)) {
        QMessageBox::warning(this, "警告", "所选文件不存在！");
        return;
    }

    disableAllInferThreads();
    m_videoThread->setVideoFilePath(fileName.toStdString());
    deployAllInferThreads();
}

