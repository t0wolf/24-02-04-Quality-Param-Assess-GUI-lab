#ifndef PARAM_PREMIUM_WIDGET_H
#define PARAM_PREMIUM_WIDGET_H

#include <QWidget>
#include <QGraphicsView>

namespace Ui {
class ParamPremiumWidget;
}

class ParamPremiumWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ParamPremiumWidget(QWidget *parent = nullptr);
    ~ParamPremiumWidget();

    int setCurrentParamsPics(QImage& qFrame);

    int setCurrentParamEvent(QString paramName)
    {
        m_paramEventName = paramName;
        return 1;
    }

    int showCurrentParamsPics();

    int setCurrentWindowTitle(const QString& title)
    {
        setWindowTitle(title);
        return 1;
    }

    inline bool isCurrentPremiumsEmpty()
    {
        return m_currrentPremiums.isNull();
    }

signals:
    void sigDeleteParam(QString paramEvent);

private:
    void adaptToView();

private slots:
    void on_deleteButton_clicked();

private:
    Ui::ParamPremiumWidget *ui;

    QImage m_currrentPremiums;

    QString m_paramEventName;

    QGraphicsScene m_scene;
};

#endif // PARAM_PREMIUM_WIDGET_H
