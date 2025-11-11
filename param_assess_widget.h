#ifndef PARAM_ASSESS_WIDGET_H
#define PARAM_ASSESS_WIDGET_H

#include <QWidget>
#include <QVariant>
#include "progress_super_thread.h"

namespace Ui {
class ParamAssessWidget;
}

class ParamAssessWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ParamAssessWidget(ProgressSuperThread* progressSuperThread, QWidget *parent = nullptr);
    ~ParamAssessWidget();

signals:
    void paramNameAvailable(const QString name);

public slots:
    void setStructureParamEvents(QVariant paramValues);

    // void setSpectrumParamEvents(QMap<QString, float> paramValues);

    // void setFuncParamEvents(QMap<QString, float> paramValues);

private:
    Ui::ParamAssessWidget *ui;

    ProgressSuperThread *m_progressSuperThread;
};

#endif // PARAM_ASSESS_WIDGET_H
