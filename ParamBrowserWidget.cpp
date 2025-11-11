#include "ParamBrowserWidget.h"

ParamBrowserWidget::ParamBrowserWidget(QWidget *parent)
	: QWidget(parent)
    , ui(new Ui::ParamBrowserWidgetClass)
    , m_fullImageDisplayLabel(new FloatingLabel())
{
	ui->setupUi(this);
    setWindowFlags(Qt::Tool | Qt::FramelessWindowHint);
    ui->imageList->installEventFilter(this);
    m_fullImageDisplayLabel->installEventFilter(this);

    qApp->installEventFilter(this);

    ui->imageList->setHorizontalScrollMode(QListWidget::ScrollPerPixel);//设置为像素滚动
    QScroller* scroller = QScroller::scroller(ui->imageList);
    scroller->grabGesture(ui->imageList, QScroller::LeftMouseButtonGesture);//设置鼠标左键拖动


    m_fullImageDisplayLabel->setMinimumSize(QSize(1200, 800));

    connect(this->ui->saveAllButton, &QPushButton::clicked, this, &ParamBrowserWidget::saveAllImages);
    connect(this->ui->deleteAllButton, &QPushButton::clicked, this, &ParamBrowserWidget::deleteAllImages);
    connect(this->ui->imageList, &QListWidget::itemDoubleClicked, this, &ParamBrowserWidget::showFullImage);
}

ParamBrowserWidget::~ParamBrowserWidget()
{}

void ParamBrowserWidget::saveAllImages() 
{
    QString fixedPath = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation) + "/SavedImages";
    QDir().mkpath(fixedPath); // 确保路径存在
    for (int i = 0; i < m_lImages.size(); ++i) {
        QString paramEventName = m_lParamNames[i];
        QString filename = QString("/image_%1_%2.png").arg(paramEventName).arg(QDateTime::currentDateTime().toString("yyyyMMddHHmmsszzz"));
        QString filePath = fixedPath + filename;

        m_lImages[i].save(filePath);
    }
}

void ParamBrowserWidget::deleteAllImages()
{
    for (int i = 0; i < m_lImages.size(); ++i)
    {
        deleteItem(i);
    }
}

void ParamBrowserWidget::showFullImage(QListWidgetItem* item)
{
    QImage image = m_mItemImages[item];
    m_fullImageDisplayLabel->setPixmap(QPixmap::fromImage(image).scaled(m_fullImageDisplayLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    m_fullImageDisplayLabel->show();
}

void ParamBrowserWidget::deleteItem(int row)
{
    QString paramEventName = m_lParamNames[row];
    ui->imageList->takeItem(row);

    if (m_lImages.size() > row)
    {
        QString fixedPath = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation) + "/DeletedImages";
        QDir().mkpath(fixedPath); // 确保路径存在
        QString filename = QString("/image_%1_%2.png").arg(paramEventName).arg(QDateTime::currentDateTime().toString("yyyyMMddHHmmsszzz"));
        QString filePath = fixedPath + filename;
        m_lImages[row].save(filePath);

        m_lImages.removeAt(row);
    }
}

int ParamBrowserWidget::updateImage(cv::Mat & image)
{
    QImage qImage = GeneralUtils::matToQImage(image);
    qImage = qImage.scaledToHeight(ui->imageList->height() - 10, Qt::SmoothTransformation);
    // 添加新图像之前检查数量
    if (m_lImages.size() >= 50) {
        // 从 QListWidget 中删除最早的项
        QListWidgetItem* firstItem = ui->imageList->takeItem(0); // 删除第一个项
        delete firstItem; // 确保释放内存

        // 从 QList 中删除对应的图像
        m_lImages.removeFirst(); // 删除第一个图像
    }

    QListWidgetItem* item = new QListWidgetItem();
    QLabel* label = new QLabel;
    label->setPixmap(QPixmap::fromImage(qImage));
    item->setSizeHint(label->sizeHint());
    ui->imageList->addItem(item);
    ui->imageList->setItemWidget(item, label);

    m_lImages.append(qImage);
	return 1;
}

int ParamBrowserWidget::updateImage(QImage& image, QString& paramEventName)
{
    if (paramEventName == "VIT" || paramEventName == "EF")
        return 0;

    QWidget* paramBrowserItemWidget = genNewParamItemWidget(image, paramEventName);

    QListWidgetItem* item = new QListWidgetItem();
    item->setSizeHint(paramBrowserItemWidget->sizeHint());
    ui->imageList->addItem(item);
    ui->imageList->setItemWidget(item, paramBrowserItemWidget);

    m_lImages.append(image);
    m_lParamNames.append(paramEventName);
    m_mItemImages.insert(item, image);
    return 1;
}

int ParamBrowserWidget::clearWidget()
{
    ui->imageList->clear();
    m_fullImageDisplayLabel->clear();
    return 1;
}

bool ParamBrowserWidget::eventFilter(QObject* obj, QEvent* event)
{
    if (event->type() == QEvent::MouseButtonPress) {
        QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
        if (mouseEvent) {
            // 检查点击位置是否在本Widget内
            if (!this->rect().contains(mouseEvent->pos())) {
                this->hide();
            }
        }
    }
    return QWidget::eventFilter(obj, event);
}


QWidget* ParamBrowserWidget::genNewParamItemWidget(QImage& image, QString& paramEventName)
{
    //QImage scaledImage = image.scaledToHeight(listWidget->height() - 30, Qt::SmoothTransformation);
    //QGraphicsPixmapItem* pixmapItem = new QGraphicsPixmapItem(QPixmap::fromImage(image.scaled(ui->imageList->width() - 10, ui->imageList->height(), Qt::KeepAspectRatio, Qt::SmoothTransformation)));
    QGraphicsPixmapItem* pixmapItem = new QGraphicsPixmapItem(QPixmap::fromImage(image));
    pixmapItem->setTransformationMode(Qt::SmoothTransformation); // 设置为平滑缩放模式
    QGraphicsScene* scene = new QGraphicsScene(this);
    scene->addItem(pixmapItem);

    QGraphicsView* view = new QGraphicsView(scene);
    view->setStyleSheet("border: none;");

    adaptToView(scene, view);

    QWidget* widget = new QWidget;
    QVBoxLayout* vLayout = new QVBoxLayout(widget);
    widget->setLayout(vLayout);

    QHBoxLayout* headerLayout = new QHBoxLayout();
    QLabel* indexLabel = new QLabel(paramEventName);
    indexLabel->setStyleSheet("color: white; background-color: rgba(0, 0, 0, 150); padding: 2px;");

    QPushButton* deleteButton = new QPushButton("×");
    deleteButton->setStyleSheet("color: white; background-color: rgba(255, 0, 0, 150); border: none; padding: 2px;");
    deleteButton->setCursor(Qt::PointingHandCursor);

    headerLayout->addWidget(indexLabel);
    headerLayout->addStretch();
    headerLayout->addWidget(deleteButton);

    vLayout->addLayout(headerLayout);
    vLayout->addWidget(view);

    connect(deleteButton, &QPushButton::clicked, this, [this, widget] {
        int row = ui->imageList->row(ui->imageList->itemAt(widget->pos()));
        deleteItem(row);
    });

    return widget;
}

int ParamBrowserWidget::adaptToView(QGraphicsScene* scene, QGraphicsView* view)
{
    QRectF rectItem = scene->itemsBoundingRect();
    QRectF rectView = view->rect();
    qreal ratioView = rectView.height() / rectView.width();
    qreal ratioItem = rectItem.height() / rectItem.width();
    if (ratioView > ratioItem)
    {
        rectItem.moveTop(rectItem.width() * ratioView - rectItem.height());
        rectItem.setHeight(rectItem.width() * ratioView);

        rectItem.setWidth(rectItem.width() * 1.5);
        rectItem.setHeight(rectItem.height() * 1.5);
    }
    else
    {
        rectItem.moveLeft(rectItem.height() / ratioView - rectItem.width());
        rectItem.setWidth(rectItem.height() / ratioView);

        rectItem.setWidth(rectItem.width() * 1.5);
        rectItem.setHeight(rectItem.height() * 1.5);
    }
    view->fitInView(rectItem, Qt::KeepAspectRatio);

    return 1;
}