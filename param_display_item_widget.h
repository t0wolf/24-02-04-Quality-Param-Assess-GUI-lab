#ifndef PARAM_DISPLAY_ITEM_WIDGET_H
#define PARAM_DISPLAY_ITEM_WIDGET_H

#include <QObject>
#include <QWidget>
#include <QString>
#include <QMouseEvent>
#include <QTimer>
#include "param_premium_widget.h"

namespace Ui {
class ParamDisplayItemWidget;
}

class ParamDisplayItemWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ParamDisplayItemWidget(QWidget *parent = nullptr);
    ~ParamDisplayItemWidget();

    int setParamValue(float fValue);

    int setParamValue(QString sValue);

    int setParamPics(QImage& qFrame);

    inline QString getName() const
    {
        return m_paramName;
    }

    void setName(QString viewName);

    void setParamValueDeleted();

    void initializeParamEvents();

private:
    Ui::ParamDisplayItemWidget* ui;

public:
    ParamPremiumWidget *m_paramPreWidget;

protected:
    void mousePressEvent(QMouseEvent* event) override;

    void mouseReleaseEvent(QMouseEvent* event) override;

    void mouseMoveEvent(QMouseEvent *event) override;

private:
    QString m_paramName;

    float m_currentValue;
    QImage m_preImage;

    QPoint m_pressPos;
    bool m_wasDragged;

};

#endif // PARAM_DISPLAY_ITEM_WIDGET_H
