#include "param_premium_widget.h"
#include "ui_param_premium_widget.h"

ParamPremiumWidget::ParamPremiumWidget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::ParamPremiumWidget)
    , m_paramEventName("")
{
    ui->setupUi(this);
    ui->premiumLabel->setScene(&m_scene);
}

ParamPremiumWidget::~ParamPremiumWidget()
{
    delete ui;
}

int ParamPremiumWidget::setCurrentParamsPics(QImage& qFrame)
{
    // QPixmap qPixmap(QPixmap::fromImage(qFrame));
    // ui->premiumLabel->setPixmap(qPixmap);
    m_currrentPremiums = qFrame;
    return 1;
}

int ParamPremiumWidget::showCurrentParamsPics()
{
    QPixmap qPixmap(QPixmap::fromImage(m_currrentPremiums));
    //ui->premiumLabel->setPixmap(qPixmap);

    //QPixmap tempPixmap = QPixmap::fromImage(qImage);
    m_scene.clear();
    m_scene.addPixmap(qPixmap);
    adaptToView();

    return 1;
}

void ParamPremiumWidget::adaptToView()
{
    QRectF rectItem = m_scene.itemsBoundingRect();
    QRectF rectView = ui->premiumLabel->rect();
    qreal ratioView = rectView.height() / rectView.width();
    qreal ratioItem = rectItem.height() / rectItem.width();
    if (ratioView > ratioItem)
    {
        rectItem.moveTop(rectItem.width() * ratioView - rectItem.height());
        rectItem.setHeight(rectItem.width() * ratioView);

        rectItem.setWidth(rectItem.width() * 1.0);
        rectItem.setHeight(rectItem.height() * 1.0);
    }
    else
    {
        rectItem.moveLeft(rectItem.height() / ratioView - rectItem.width());
        rectItem.setWidth(rectItem.height() / ratioView);

        rectItem.setWidth(rectItem.width() * 1.0);
        rectItem.setHeight(rectItem.height() * 1.0);
    }
    ui->premiumLabel->fitInView(rectItem, Qt::KeepAspectRatio);
}

void ParamPremiumWidget::on_deleteButton_clicked()
{
    //ui->premiumLabel->clear();
    m_scene.clear();
    m_scene.addPixmap(QPixmap());

    m_currrentPremiums = QImage();

    emit sigDeleteParam(m_paramEventName);
}

