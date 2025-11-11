#ifndef DISPLAY_STREAM_WIDGET_H
#define DISPLAY_STREAM_WIDGET_H
#pragma execution_character_set("utf-8")

#include <QWidget>
#include <opencv2/opencv.hpp>
#include <QGraphicsView>
#include <QFileDialog>
#include "video_stream_thread.h"
#include "models_inference_thread.h"
#include "quality_display_widget.h"
#include "param_display_widget.h"
#include "progress_super_thread.h"
#include "process_threads/InfoExtractThread.h"
#include "process_threads/ROIScaleDetThread.h"
#include "DataBuffer.h"
#include "config_parse.h"
#include "ParamAssessHTMLShower.h"

namespace Ui {
class DisplayStreamWidget;
}

class DisplayStreamWidget : public QWidget
{
    Q_OBJECT

public:
    explicit DisplayStreamWidget(QualityDisplayWidget* qualityDisplayWidget,
        ParamDisplayWidget* paramDisplayWidget,
        ProgressSuperThread* progressThread,
        ConfigParse* config = nullptr,
        QWidget *parent = nullptr);
    ~DisplayStreamWidget();

signals:


private slots:
    void on_beginButton_clicked();
    void displayFrame(const QImage qImage);

    void displayDebugText(QString text);

    //void on_equalButton_clicked();

    void on_readVideoButton_clicked();

private:
    void adaptToView();

    int inputFramesToThreads(const QImage& qImage);

    int deployAllInferThreads();

    int disableAllInferThreads();
    //ROIScaleDetThread* m_roiScaleThread;


public:
    Ui::DisplayStreamWidget *ui;
    QualityDisplayWidget* m_qualityDisplayWidget;
    ParamDisplayWidget* m_paramDisplayWidget;

    ROIScaleDetThread* m_roiScaleThread;
    VideoStreamThread *m_videoThread;
    ModelsInferenceThread *m_modelsInferThread;
    InfoExtractThread* m_infoExtractThread;

    ParamAssessHTMLShower* m_paramAssessHtmlShower;

    QVector<cv::Rect> m_drawRects;
    QVector<QString> m_drawLabels;

    // ParamAssessWidget *m_paramAssessWidget;
    // ProgressSuperThread *m_progressThread;

    QGraphicsScene m_scene;
    QGraphicsTextItem* m_textItem;

    QString m_debugText;
    QString m_debugTextbuffer;
    QStringList m_debugTexts;

    DataBuffer* m_roiDataBuffer;
};

#endif // DISPLAY_STREAM_WIDGET_H
