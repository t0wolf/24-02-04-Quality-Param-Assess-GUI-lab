#ifndef QUALITY_CONTROL_WIDGET_H
#define QUALITY_CONTROL_WIDGET_H

#include <QWidget>
#include <QVariant>
#include <QVector>
#include "models_inference_thread.h"
#include "rapidjson/document.h"
#include "type_registering.h"

namespace Ui {
class QualityControlWidget;
}

class QualityControlWidget : public QWidget
{
    Q_OBJECT

public:
    explicit QualityControlWidget(QWidget *parent = nullptr);
    ~QualityControlWidget();

    int parseQualityScores(std::string& viewName, singleResult& videoResult);

private:
    int parseJSONFile(std::string jsonPath);

public slots:
    void setCurrentViewName(const QString viewName);

    void setCurrentViewQualityScores(QVariant currResult);

private:
    Ui::QualityControlWidget *ui;
    std::string m_status;
    QString m_viewName;
public:
    rapidjson::Document m_viewRulesTotal;
};

#endif // QUALITY_CONTROL_WIDGET_H
