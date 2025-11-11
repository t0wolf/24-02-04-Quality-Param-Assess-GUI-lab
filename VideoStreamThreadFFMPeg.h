#pragma once

#include <QObject>
#include <QImage>
#include <QMutex>
#include <QThread>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}


class VideoStreamThreadFFMPeg  : public QThread
{
	Q_OBJECT

public:
	VideoStreamThreadFFMPeg(QObject *parent);
	~VideoStreamThreadFFMPeg();
};
