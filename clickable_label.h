#ifndef CLICKABLELABEL_H
#define CLICKABLELABEL_H

#include <QLabel>
#include <QObject>
#include <QWidget>
#include <QTimer>
#include <QThread>
#include <QVector>
#include <QApplication>
#include <QScreen>

#include "opencv2/opencv.hpp"
#include "quality_detail_widget.h"
#include "quality_detail_play_thread.h"


class ClickableLabel : public QLabel
{
    Q_OBJECT
public:
    explicit ClickableLabel(QWidget* parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());

    void setViewName(const QString viewName);

    ~ClickableLabel();

signals:
    void mouseClicked();

    // void frameAvailable(QImage& image);

public:
    QVector<cv::Mat> m_videoClipBuffer;

    QualityDetailWidget* m_qualityDetailWidget;

public slots:
    void slotDisplayQualityDetail();

    void setPlayThreadExit();

    void setInitialize();

    void setQualityScores(QVariant qVResults);

protected:
    void mousePressEvent(QMouseEvent* event) override;

private:
    QualityDetailPlayThread* m_detailPlayThread;

    QString m_viewName;

};

#endif // CLICKABLELABEL_H
