#ifndef QUALITY_DETAIL_WIDGET_H
#define QUALITY_DETAIL_WIDGET_H

#include <QWidget>
#include <QString>
#include "rapidjson/document.h"
#include <QGraphicsView>
// #include "clickable_label.h"

namespace Ui {
class QualityDetailWidget;
}

// class QualityDetailWidget : public QWidget
// {
//     Q_OBJECT

// public:
//     explicit QualityDetailWidget(QWidget *parent = nullptr, ClickableLabel* clickableLabel = nullptr);
//     ~QualityDetailWidget();

// public slots:
//     void slotsDisplayDetailed();

// private:
//     Ui::QualityDetailWidget *ui;

//     ClickableLabel* m_clickableLabel;
// };

class QualityDetailWidget : public QWidget
{
    Q_OBJECT

public:
    explicit QualityDetailWidget(QWidget *parent = nullptr, QString viewName = "");
    ~QualityDetailWidget();
    inline void setViewName(const QString viewName)
    {
        m_viewName = viewName;
    }

    void setQualityRules(rapidjson::Value& qualityRules);

    void clearLabelTable();

    float setQualityScores(QVariant currResult);

private:
    int parseJSONFile();

    void adaptToView();

public:
    rapidjson::Value m_qualityRules;

signals:
    void deleteAvailable(const QString viewName);

    void clearLabelAvailable();

public slots:
    void setSavedVideoFrame(QImage qImage);

    void setLabelTableClear();

    // void setQualityScores(QVariant currResult);

private slots:
    void on_pushButton_clicked();

private:
    Ui::QualityDetailWidget *ui;
    QString m_viewName;
    QGraphicsScene m_scene;
};

#endif // QUALITY_DETAIL_WIDGET_H
