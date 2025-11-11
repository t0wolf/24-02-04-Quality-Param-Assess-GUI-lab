#pragma once

#include <QWidget>
#include <QLabel>
#include <QImage>
#include <QPixmap>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QListWidget>
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QListWidgetItem>
#include <QStandardPaths>
#include <QDateTime>
#include <QDir>
#include <QMouseEvent>
#include <QScroller>
#include "general_utils.h"
#include "ui_ParamBrowserWidget.h"

class FloatingLabel : public QLabel
{
	Q_OBJECT

public:
	explicit FloatingLabel(QWidget* parent = nullptr) : QLabel(parent) {}

protected:
	void mousePressEvent(QMouseEvent* event) override
	{
		// 点击 QLabel 时隐藏它
		if (isVisible())
		{
			hide();
		}
	}
};


class ParamBrowserWidget : public QWidget
{
	Q_OBJECT

public:
	ParamBrowserWidget(QWidget *parent = nullptr);
	~ParamBrowserWidget();

	int updateImage(cv::Mat& image);

	int updateImage(QImage& image, QString& paramEventName);

	int clearWidget();

protected:
	bool eventFilter(QObject* obj, QEvent* event) override;

private:
	QWidget* genNewParamItemWidget(QImage& image, QString& paramEventName);

	int adaptToView(QGraphicsScene* scene, QGraphicsView* view);

public slots:
	void deleteItem(int row);

	void saveAllImages();

	void deleteAllImages();

	void showFullImage(QListWidgetItem* item);

private:
	Ui::ParamBrowserWidgetClass* ui;

	FloatingLabel* m_fullImageDisplayLabel;

	QList<QImage> m_lImages;
	QMap<QListWidgetItem*, QImage> m_mItemImages;
	QList<QString> m_lParamNames;
};
