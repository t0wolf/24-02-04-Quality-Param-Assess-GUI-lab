QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    DarkStyle.cpp \
    clickable_label.cpp \
    config_parse.cpp \
    display_stream_widget.cpp \
    framelesswindow/framelesswindow.cpp \
    framelesswindow/windowdragger.cpp \
    general_utils.cpp \
    logger.cpp \
    main.cpp \
    mainwindow.cpp \
    models_inference_thread.cpp \
    param_assess_thread.cpp \
    param_assess_widget.cpp \
    param_assesser.cpp \
    # param_assessment/Quality-Param.cpp \
    param_assessment/aad_assess.cpp \
    param_assessment/aao_assess.cpp \
    param_assessment/aao_segment_inferer.cpp \
    param_assessment/asd_and_sjd_assess.cpp \
    param_assessment/assess_utils.cpp \
    param_assessment/avad_assess.cpp \
    param_assessment/avad_keypoints_inferer.cpp \
    param_assessment/ef_assess.cpp \
    param_assessment/ef_segment_inferer.cpp \
    param_assessment/ef_video_inferer.cpp \
    param_assessment/image_process.cpp \
    param_assessment/ivs_and_pw_assess.cpp \
    param_assessment/laad_assess.cpp \
    param_assessment/logger.cpp \
    param_assessment/plax_params_assess.cpp \
    param_assessment/segment_infer_base.cpp \
    param_display_item_widget.cpp \
    param_display_widget.cpp \
    param_premium_widget.cpp \
    process_threads/quality_control_thread.cpp \
    progress_super_thread.cpp \
    quality_control/AngleCalculationA4C.cpp \
    quality_control/EchoQuality.cpp \
    quality_control/EchoQualityAssessmentA4C.cpp \
    quality_control/EchoQualityAssessmentPSAXA.cpp \
    quality_control/EchoQualityAssessmentPSAXGV.cpp \
    quality_control/EchoQualityAssessmentPSAXMV.cpp \
    quality_control/GainClassification.cpp \
    quality_control/KeyframeDetection.cpp \
    quality_control/RoIExtraction.cpp \
    quality_control/StructureDetection.cpp \
    quality_control/angleinfer_a4c.cpp \
    # quality_control/common/getOptions.cpp \
    # quality_control/common/sampleEngines.cpp \
    # quality_control/common/sampleInference.cpp \
    # quality_control/common/sampleOptions.cpp \
    # quality_control/common/sampleReporting.cpp \
    # quality_control/common/sampleUtils.cpp \
    # quality_control/common/windows/getopt.c \
    quality_control/cplusfuncs.cpp \
    quality_control/echo_quality_control.cpp \
    quality_control/gaininfer.cpp \
    quality_control/integrity_classification.cpp \
    quality_control/keyframe_det_inferer.cpp \
    quality_control/keyframe_detector.cpp \
    quality_control/keyframeinfer.cpp \
    quality_control/opencvfuncs.cpp \
    quality_control/quality_control.cpp \
    quality_control/quality_utils.cpp \
    quality_control/roi_detection.cpp \
    quality_control/structureinfer.cpp \
    # quality_control_thread.cpp \
    quality_control_widget.cpp \
    quality_detail_play_thread.cpp \
    quality_detail_widget.cpp \
    quality_display_widget.cpp \
    video_stream_thread.cpp \
    view_classification_inferer.cpp \
    view_cls_inferer.cpp \
    view_progress_widget.cpp

HEADERS += \
    DarkStyle.h \
    clickable_label.h \
    config_parse.h \
    display_stream_widget.h \
    framelesswindow/framelesswindow.h \
    framelesswindow/windowdragger.h \
    general_utils.h \
    logger.h \
    logging.h \
    mainwindow.h \
    models_inference_thread.h \
    param_assess_thread.h \
    param_assess_widget.h \
    param_assesser.h \
    param_assessment/aad_assess.h \
    param_assessment/aao_assess.h \
    param_assessment/aao_segment_inferer.h \
    param_assessment/asd_and_sjd_access.h \
    param_assessment/assess_utils.h \
    param_assessment/avad_assess.h \
    param_assessment/avad_keypoints_inferer.h \
    param_assessment/ef_assess.h \
    param_assessment/ef_segment_inferer.h \
    param_assessment/ef_video_inferer.h \
    param_assessment/image_process.h \
    param_assessment/ivs_and_pw_assess.h \
    param_assessment/laad_assess.h \
    param_assessment/logger.h \
    param_assessment/logging.h \
    param_assessment/plax_params_assess.h \
    param_assessment/segment_infer_base.h \
    param_display_item_widget.h \
    param_display_widget.h \
    param_premium_widget.h \
    process_threads/quality_control_thread.h \
    progress_super_thread.h \
    quality_control/AngleCalculationA4C.h \
    quality_control/EchoQuality.h \
    quality_control/EchoQualityAssessmentA4C.h \
    quality_control/EchoQualityAssessmentPSAXA.h \
    quality_control/EchoQualityAssessmentPSAXGV.h \
    quality_control/EchoQualityAssessmentPSAXMV.h \
    quality_control/GainClassification.h \
    quality_control/KeyframeDetection.h \
    quality_control/RoIExtraction.h \
    quality_control/StructureDetection.h \
    quality_control/angleinfer_a4c.h \
    # quality_control/common/BatchStream.h \
    # quality_control/common/EntropyCalibrator.h \
    # quality_control/common/ErrorRecorder.h \
    # quality_control/common/argsParser.h \
    # quality_control/common/buffers.h \
    # quality_control/common/common.h \
    # quality_control/common/getOptions.h \
    # quality_control/common/half.h \
    # quality_control/common/parserOnnxConfig.h \
    # quality_control/common/safeCommon.h \
    # quality_control/common/sampleConfig.h \
    # quality_control/common/sampleDevice.h \
    # quality_control/common/sampleEngines.h \
    # quality_control/common/sampleInference.h \
    # quality_control/common/sampleOptions.h \
    # quality_control/common/sampleReporting.h \
    # quality_control/common/sampleUtils.h \
    # quality_control/common/windows/getopt.h \
    quality_control/buffers.h \
    quality_control/cplusfuncs.h \
    quality_control/echo_quality_control.h \
    quality_control/gaininfer.h \
    quality_control/integrity_classification.h \
    quality_control/iqaparams.h \
    quality_control/keyframe_det_inferer.h \
    quality_control/keyframe_detector.h \
    quality_control/keyframeinfer.h \
    quality_control/opencvfuncs.h \
    quality_control/quality_control.h \
    quality_control/quality_utils.h \
    quality_control/roi_detection.h \
    quality_control/structureinfer.h \
    quality_control/typemappings.h \
    # quality_control_thread.h \
    quality_control_widget.h \
    quality_detail_play_thread.h \
    quality_detail_widget.h \
    quality_display_widget.h \
    rapidjson/allocators.h \
    rapidjson/cursorstreamwrapper.h \
    rapidjson/document.h \
    rapidjson/encodedstream.h \
    rapidjson/encodings.h \
    rapidjson/error/en.h \
    rapidjson/error/error.h \
    rapidjson/filereadstream.h \
    rapidjson/filewritestream.h \
    rapidjson/fwd.h \
    rapidjson/internal/biginteger.h \
    rapidjson/internal/clzll.h \
    rapidjson/internal/diyfp.h \
    rapidjson/internal/dtoa.h \
    rapidjson/internal/ieee754.h \
    rapidjson/internal/itoa.h \
    rapidjson/internal/meta.h \
    rapidjson/internal/pow10.h \
    rapidjson/internal/regex.h \
    rapidjson/internal/stack.h \
    rapidjson/internal/strfunc.h \
    rapidjson/internal/strtod.h \
    rapidjson/internal/swap.h \
    rapidjson/istreamwrapper.h \
    rapidjson/memorybuffer.h \
    rapidjson/memorystream.h \
    rapidjson/msinttypes/inttypes.h \
    rapidjson/msinttypes/stdint.h \
    rapidjson/ostreamwrapper.h \
    rapidjson/pointer.h \
    rapidjson/prettywriter.h \
    rapidjson/rapidjson.h \
    rapidjson/reader.h \
    rapidjson/schema.h \
    rapidjson/stream.h \
    rapidjson/stringbuffer.h \
    rapidjson/uri.h \
    rapidjson/writer.h \
    typemapping.h \
    video_stream_thread.h \
    view_classification_inferer.h \
    view_cls_inferer.h \
    view_progress_widget.h

FORMS += \
    display_stream_widget.ui \
    framelesswindow/framelesswindow.ui \
    mainwindow.ui \
    # mainwindow_copy.ui \
    param_assess_widget.ui \
    param_assess_widget_copy.ui \
    param_display_item_widget.ui \
    param_display_widget.ui \
    param_premium_widget.ui \
    quality_control_widget.ui \
    quality_detail_widget.ui \
    quality_display_widget.ui \
    view_progress_widget.ui \
    view_progress_widget_copy.ui


msvc {
    QMAKE_CFLAGS += /utf-8
    QMAKE_CXXFLAGS += /utf-8
}

INCLUDEPATH += D:/local_install/opencv-4.5.5/opencv/build/include/opencv2 \
LIBS += D:/local_install/opencv-4.5.5/opencv/build/include

LIBS += -lD:/local_install/opencv-4.5.5/opencv/build/x64/vc15/lib/opencv_world455d

INCLUDEPATH += D:/local_install/CUDA/v11.6/include
LIBS += -LD:/local_install/CUDA/v11.6/lib/x64
LIBS += -lcuda \
        -lcudart

INCLUDEPATH += D:/local_install/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/TensorRT-8.6.1.6/include
LIBS += -LD:/local_install/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/TensorRT-8.6.1.6/lib
LIBS += -lnvinfer \
        -lnvparsers \
        -lnvinfer_plugin \
        -lnvonnxparser

INCLUDEPATH += D:/local_install/yaml-cpp/include
LIBS += -lD:/local_install/yaml-cpp/lib/yaml-cppd

INCLUDEPATH += D/local_install/rapidjson/include/rapidjson
# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

DISTFILES += \
    darkstyle/darkstyle.qss \
    darkstyle/icon_branch_closed.png \
    darkstyle/icon_branch_end.png \
    darkstyle/icon_branch_more.png \
    darkstyle/icon_branch_open.png \
    darkstyle/icon_checkbox_checked.png \
    darkstyle/icon_checkbox_checked_disabled.png \
    darkstyle/icon_checkbox_checked_pressed.png \
    darkstyle/icon_checkbox_indeterminate.png \
    darkstyle/icon_checkbox_indeterminate_disabled.png \
    darkstyle/icon_checkbox_indeterminate_pressed.png \
    darkstyle/icon_checkbox_unchecked.png \
    darkstyle/icon_checkbox_unchecked_disabled.png \
    darkstyle/icon_checkbox_unchecked_pressed.png \
    darkstyle/icon_close.png \
    darkstyle/icon_radiobutton_checked.png \
    darkstyle/icon_radiobutton_checked_disabled.png \
    darkstyle/icon_radiobutton_checked_pressed.png \
    darkstyle/icon_radiobutton_unchecked.png \
    darkstyle/icon_radiobutton_unchecked_disabled.png \
    darkstyle/icon_radiobutton_unchecked_pressed.png \
    darkstyle/icon_restore.png \
    darkstyle/icon_undock.png \
    darkstyle/icon_vline.png \
    quality_control/common/dumpTFWts.py

RESOURCES += \
    darkstyle.qrc \
    framelesswindow.qrc \
    quality_param_system.qrc
