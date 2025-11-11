#ifndef QUALITY_DISPLAY_WIDGET_H
#define QUALITY_DISPLAY_WIDGET_H

#include <QWidget>
#include <QLabel>
#include <QPixmap>
#include <QBuffer>
#include <QVariant>
#include <QMap>
#include <QPropertyAnimation>

#include <fstream>
#include <opencv2/opencv.hpp>

// #include "quality_detail_widget.h"
#include "clickable_label.h"
#include "progress_super_thread.h"
#include "config_parse.h"

// Q_DECLARE_METATYPE(QVector<cv::Mat>)
Q_DECLARE_METATYPE(cv::Mat)

const QString GREEN_COLOR = "#43A047";
const QString RED_COLOR = "#EF5350";
const QString FONT_SIZE = "12px";

namespace Ui {
class QualityDisplayWidget;
}

class QualityDisplayWidget : public QWidget
{
    Q_OBJECT

public:
    explicit QualityDisplayWidget(QWidget *parent = nullptr, ProgressSuperThread *progressThread = nullptr, ConfigParse* configFile = nullptr);
    ~QualityDisplayWidget();

    int labelReInit();

signals:
    void exitThreadAvailable();

    void labelInitalizeAvailable();

    void qualityScoresAvailable(QVariant qVResults);

public slots:
    void setCurrentViewName(const QString viewName);

    void setCurrentViewNameImage(const QString viewName, QImage qImage);

    void setCurrentViewNameVideo(const QString viewName, QVariant qVar);

    void setCurrentViewQualityControlled(const QString viewName, QVariant qVideoClips, QVariant qVResult);

    void setLabelInitialize(const QString viewName);

    void setViewUncomplete(const QString viewName);

    //void setQualityScores(QString viewName, QVariant qVResult);

private:
    int setViewNameVideo(const QString& viewName, QVector<cv::Mat>& vVideoCLips, QString& color);

    void setQualityScores(QString viewName, QVariant qVResult);

    QPair<bool, bool> judgeQualityScores(const QString& viewName, QVector<float>& vResults);

    inline void initializeBestScores()
    {
        m_viewBestQualityScores = {
            {"A2C", -100.0},
            {"A3C", -100.0},
            {"A4C", -100.0},
            {"A5C", -100.0},
            {"PLAX", -100.0},
            {"PSAXA", -100.0},
            {"PSAXGV", -100.0},
            {"PSAXPM", -100.0},
            {"PSAXMV", -100.0},
        };
    }

private:
    Ui::QualityDisplayWidget *ui;

    ProgressSuperThread *m_progressThread;

    std::vector<QString> m_viewNameMapping = { "A2C", "A3C" ,"A4C" , "A5C", "PLAX", "PSAXA", "PSAXGV", "PSAXMV", "PSAXPM" };

    QMap<QString, bool> m_viewCompleteMap = {
        {"A2C", false},
        {"A3C", false},
        {"A4C", false},
        {"A5C", false},
        {"PLAX", false},
        {"PSAXA", false},
        {"PSAXGV", false},
        {"PSAXPM", false},
        {"PSAXMV", false},
    };

    QMap<QString, float> m_viewQualityThresh = {
        {"A2C", 8.0},
        {"A3C", 8.0},
        {"A4C", 8.0},
        {"A5C", 8.0},
        {"PLAX", 8.0},
        {"PSAXA", 4.0},
        {"PSAXGV", 8.0},
        {"PSAXPM", 8.0},
        {"PSAXMV", 8.0},
    };

    QMap<QString, float> m_viewBestQualityScores;

    ClickableLabel* m_currentActiveLabel;

    rapidjson::Document m_viewRulesTotal;

    // QVector<cv::Mat> m_a2cVideoClip;
    // QVector<cv::Mat> m_a3cVideoClip;
    // QVector<cv::Mat> m_a4cVideoClip;
    // QVector<cv::Mat> m_a5cVideoClip;
    // QVector<cv::Mat> m_plaxVideoClip;
    // QVector<cv::Mat> m_psaxaVideoClip;
    // QVector<cv::Mat> m_psaxgvVideoClip;
    // QVector<cv::Mat> m_psaxpmVideoClip;
    // QVector<cv::Mat> m_psaxmvVideoClip;

private:
    int initialize();

    int setApperance(QLabel* label, QPixmap& qImage, const QString& dispText, const QString color);

    int setApperance(QLabel* label, const QString& dispText);

    int setApperance(QLabel* label, const QString& dispText, const QString color);

    ClickableLabel* getViewLabel(const QString& viewName);

    void setActiveLabel(ClickableLabel* label);

    int parseJSONFile(std::string jsonPath);

};

#endif // QUALITY_DISPLAY_WIDGET_H
