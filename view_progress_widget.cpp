#include "view_progress_widget.h"
#include "ui_view_progress_widget.h"

ViewProgressWidget::ViewProgressWidget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::ViewProgressWidget)
    , m_progressSuperThread(new ProgressSuperThread(this))
{
    ui->setupUi(this);

    connect(m_progressSuperThread, &ProgressSuperThread::uiProgressUpdateAvailable, this, &ViewProgressWidget::setProgressListUpdate);
    m_progressSuperThread->start();
}

ViewProgressWidget::~ViewProgressWidget()
{
    delete ui;
}

void ViewProgressWidget::setProgressListUpdate(const QString name)
{
    if (m_vApicalViewNames.contains(name))
    {
        int viewIdx = m_vApicalViewNames.indexOf(name);
		ui->apicalProgressList->item(viewIdx)->setBackgroundColor(Qt::green);
	}
	else if (m_vAxisViewNames.contains(name))
	{
		int viewIdx = m_vAxisViewNames.indexOf(name);
        ui->shortProgressList->item(viewIdx)->setBackgroundColor(Qt::green);
	}
    else
    {
		// qDebug() << "ViewProgressWidget::updateProgressList: " << name << " is not a valid view name";
	}
}