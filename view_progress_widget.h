#ifndef VIEW_PROGRESS_WIDGET_H
#define VIEW_PROGRESS_WIDGET_H

#include <QWidget>
#include <qmap.h>
#include <qvector.h>
#include "progress_super_thread.h"

namespace Ui {
class ViewProgressWidget;
}

class ViewProgressWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ViewProgressWidget(QWidget *parent = nullptr);
    ~ViewProgressWidget();

public slots:
    void setProgressListUpdate(const QString name);

private:
    Ui::ViewProgressWidget *ui;
  
    ProgressSuperThread* m_progressSuperThread;

    QVector<QString> m_vViewNames = { "A2C", "A3C" ,"A4C" , "A5C", "PLAX", "PSAXA", "PSAXGV", "PSAXMV", "PSAXPM" };
    QVector<QString> m_vApicalViewNames = { "A2C", "A3C" ,"A4C" , "A5C" };
    QVector<QString> m_vAxisViewNames = { "PLAX", "PSAXA", "PSAXGV", "PSAXMV", "PSAXPM" };
};

#endif // VIEW_PROGRESS_WIDGET_H
