cmake_minimum_required(VERSION 3.28)
project(24_02_04_Quality_Param_Assess_GUI)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(QT_VERSION 5)


set(CMAKE_PREFIX_PATH "D:/local_install/Qt-5.15/5.15.2/msvc2019_64")

find_package(Qt5 COMPONENTS
        Core
        Gui
        Widgets
        REQUIRED)
set(OpenCV_DIR "D:/software/opencv/opencv/build")  # 修改为你的实际路径
find_package(OpenCV REQUIRED)

set(YAML_CPP_INSTALL_PATH "C:/Libs/yaml-cpp")
set(TENSORRT_ROOT "D:/software/TensorRT-8.4.1.5")
set(FFMPEG_ROOT "C:/Libs/ffmpeg-8.0-full_build-shared") # **请修改为您的实际安装路径**


include_directories(.)
include_directories(debug)
include_directories(framelesswindow)
include_directories(info_extraction)
include_directories(ocr_inference)
include_directories(param_assessment)
include_directories(process_threads)
include_directories(quality_control)
include_directories(quality_control/common)
include_directories(quality_control/common/windows)
include_directories(rapidjson)
include_directories(rapidjson/error)
include_directories(rapidjson/internal)
include_directories(rapidjson/msinttypes)
include_directories(SYSTEM ${YAML_CPP_INSTALL_PATH}/include)
include_directories(SYSTEM ${TENSORRT_ROOT}/include)
include_directories(SYSTEM ${FFMPEG_ROOT}/include)

find_library(YAML_CPP_LIBRARY NAMES yaml-cpp PATHS ${YAML_CPP_INSTALL_PATH}/lib REQUIRED)

add_executable(24_02_04_Quality_Param_Assess_GUI
#        debug/qrc_darkstyle.cpp
#        debug/qrc_framelesswindow.cpp
#        debug/qrc_quality_param_system.cpp
#        framelesswindow/framelesswindow.cpp
#        framelesswindow/framelesswindow.h
#        framelesswindow/windowdragger.cpp
#        framelesswindow/windowdragger.h
        info_extraction/InfoExtractor.cpp
        info_extraction/InfoExtractor.h
        info_extraction/ScaleExtractor.cpp
        info_extraction/ScaleExtractor.h
        ocr_inference/OCRDet.cpp
        ocr_inference/OCRDet.h
        ocr_inference/OCRInferer.cpp
        ocr_inference/OCRInferer.h
        ocr_inference/OCRRec.cpp
        ocr_inference/OCRRec.h
        param_assessment/aad_assess.cpp
        param_assessment/aad_assess.h
        param_assessment/aao_assess.cpp
        param_assessment/aao_assess.h
        param_assessment/aao_segment_inferer.cpp
        param_assessment/aao_segment_inferer.h
        param_assessment/asd_and_sjd_access.h
        param_assessment/asd_and_sjd_assess.cpp
        param_assessment/assess_utils.cpp
        param_assessment/assess_utils.h
        param_assessment/avad_assess.cpp
        param_assessment/avad_assess.h
        param_assessment/avad_keypoints_inferer.cpp
        param_assessment/avad_keypoints_inferer.h
        param_assessment/DoSpecParamsAssess.cpp
        param_assessment/DoSpecParamsAssess.h
        param_assessment/ef_assess.cpp
        param_assessment/ef_assess.h
        param_assessment/ef_segment_inferer.cpp
        param_assessment/ef_segment_inferer.h
        param_assessment/ef_video_inferer.cpp
        param_assessment/ef_video_inferer.h
        param_assessment/fvav_assess.cpp
        param_assessment/fvav_assess.h
        param_assessment/image_process.cpp
        param_assessment/image_process.h
        param_assessment/ivs_and_pw_assess.cpp
        param_assessment/ivs_and_pw_assess.h
        param_assessment/laad_assess.cpp
        param_assessment/laad_assess.h
        param_assessment/logger.cpp
        param_assessment/logger.h
        param_assessment/logging.h
        param_assessment/mvea_assess.cpp
        param_assessment/mvea_assess.h
        param_assessment/mvea_keypoints_inferer.cpp
        param_assessment/mvea_keypoints_inferer.h
        param_assessment/param_roi_detection.cpp
        param_assessment/param_roi_detection.h
        param_assessment/plax_params_assess.cpp
        param_assessment/plax_params_assess.h
        param_assessment/pv_assess.cpp
        param_assessment/pv_assess.h
        param_assessment/pv_keypoints_inferer.cpp
        param_assessment/pv_keypoints_inferer.h
        param_assessment/segment_infer_base.cpp
        param_assessment/segment_infer_base.h
        param_assessment/tdi_ivs_assess.cpp
        param_assessment/tdi_ivs_assess.h
        param_assessment/tdi_ivs_keypoints_inferer.cpp
        param_assessment/tdi_ivs_keypoints_inferer.h
        param_assessment/tdi_mvlw_assess.cpp
        param_assessment/tdi_mvlw_assess.h
        param_assessment/tr_assess.cpp
        param_assessment/tr_assess.h
        process_threads/InfoExtractThread.cpp
        process_threads/InfoExtractThread.h
        process_threads/ParamAssessThread.cpp
        process_threads/ParamAssessThread.h
        process_threads/quality_control_thread.cpp
        process_threads/quality_control_thread.h
        quality_control/common/windows/getopt.c
        quality_control/common/windows/getopt.h
        quality_control/common/argsParser.h
        quality_control/common/BatchStream.h
        quality_control/common/buffers.h
        quality_control/common/common.h
        quality_control/common/EntropyCalibrator.h
        quality_control/common/ErrorRecorder.h
        quality_control/common/getOptions.cpp
        quality_control/common/getOptions.h
        quality_control/common/half.h
        quality_control/common/logging.h
        quality_control/common/parserOnnxConfig.h
        quality_control/common/safeCommon.h
        quality_control/common/sampleConfig.h
        quality_control/common/sampleDevice.h
        quality_control/common/sampleEngines.cpp
        quality_control/common/sampleEngines.h
        quality_control/common/sampleInference.cpp
        quality_control/common/sampleInference.h
        quality_control/common/sampleOptions.cpp
        quality_control/common/sampleOptions.h
        quality_control/common/sampleReporting.cpp
        quality_control/common/sampleReporting.h
        quality_control/common/sampleUtils.cpp
        quality_control/common/sampleUtils.h
        quality_control/AngleCalculationA4C.cpp
        quality_control/AngleCalculationA4C.h
        quality_control/AngleCalculationPLAX.cpp
        quality_control/AngleCalculationPLAX.h
        quality_control/angleinfer_a4c.cpp
        quality_control/angleinfer_a4c.h
        quality_control/angleinfer_plax.cpp
        quality_control/angleinfer_plax.h
        quality_control/buffers.h
        quality_control/cplusfuncs.cpp
        quality_control/cplusfuncs.h
        quality_control/echo_quality_control.cpp
        quality_control/echo_quality_control.h
        quality_control/EchoQuality.cpp
        quality_control/EchoQuality.h
        quality_control/EchoQualityAssessmentA2C.cpp
        quality_control/EchoQualityAssessmentA2C.h
        quality_control/EchoQualityAssessmentA3C.cpp
        quality_control/EchoQualityAssessmentA3C.h
        quality_control/EchoQualityAssessmentA4C.cpp
        quality_control/EchoQualityAssessmentA4C.h
        quality_control/EchoQualityAssessmentA5C.cpp
        quality_control/EchoQualityAssessmentA5C.h
        quality_control/EchoQualityAssessmentPLAX.cpp
        quality_control/EchoQualityAssessmentPLAX.h
        quality_control/EchoQualityAssessmentPSAXA.cpp
        quality_control/EchoQualityAssessmentPSAXA.h
        quality_control/EchoQualityAssessmentPSAXGV.cpp
        quality_control/EchoQualityAssessmentPSAXGV.h
        quality_control/EchoQualityAssessmentPSAXMV.cpp
        quality_control/EchoQualityAssessmentPSAXMV.h
        quality_control/EchoQualityAssessmentPSAXPM.cpp
        quality_control/EchoQualityAssessmentPSAXPM.h
        quality_control/GainClassification.cpp
        quality_control/GainClassification.h
        quality_control/gaininfer.cpp
        quality_control/gaininfer.h
        quality_control/integrity_classification.cpp
        quality_control/integrity_classification.h
        quality_control/iqaparams.h
        quality_control/keyframe_det_inferer.cpp
        quality_control/keyframe_det_inferer.h
        quality_control/keyframe_detector.cpp
        quality_control/keyframe_detector.h
        quality_control/KeyframeDetection.cpp
        quality_control/KeyframeDetection.h
        quality_control/keyframeinfer.cpp
        quality_control/keyframeinfer.h
        quality_control/opencvfuncs.cpp
        quality_control/opencvfuncs.h
     
        
        quality_control/quality_utils.cpp
        quality_control/quality_utils.h
        quality_control/roi_detection.cpp
        quality_control/roi_detection.h
        quality_control/RoIExtraction.cpp
        quality_control/RoIExtraction.h
        quality_control/StructureDetection.cpp
        quality_control/StructureDetection.h
        quality_control/structureinfer.cpp
        quality_control/structureinfer.h
        quality_control/typemappings.h
        rapidjson/error/en.h
        rapidjson/error/error.h
        rapidjson/internal/biginteger.h
        rapidjson/internal/clzll.h
        rapidjson/internal/diyfp.h
        rapidjson/internal/dtoa.h
        rapidjson/internal/ieee754.h
        rapidjson/internal/itoa.h
        rapidjson/internal/meta.h
        rapidjson/internal/pow10.h
        rapidjson/internal/regex.h
        rapidjson/internal/stack.h
        rapidjson/internal/strfunc.h
        rapidjson/internal/strtod.h
        rapidjson/internal/swap.h
        rapidjson/msinttypes/inttypes.h
        rapidjson/msinttypes/stdint.h
        rapidjson/allocators.h
        rapidjson/cursorstreamwrapper.h
        rapidjson/document.h
        rapidjson/encodedstream.h
        rapidjson/encodings.h
        rapidjson/filereadstream.h
        rapidjson/filewritestream.h
        rapidjson/fwd.h
        rapidjson/istreamwrapper.h
        rapidjson/memorybuffer.h
        rapidjson/memorystream.h
        rapidjson/ostreamwrapper.h
        rapidjson/pointer.h
        rapidjson/prettywriter.h
        rapidjson/rapidjson.h
        rapidjson/reader.h
        rapidjson/schema.h
        rapidjson/stream.h
        rapidjson/stringbuffer.h
        rapidjson/uri.h
        rapidjson/writer.h
        clickable_label.cpp
        clickable_label.h
        config_parse.cpp
        config_parse.h
        DarkStyle.cpp
        DarkStyle.h
        display_stream_widget.cpp
        display_stream_widget.h
        general_utils.cpp
        general_utils.h
        logger.cpp
        logger.h
        logging.h
        main.cpp
        mainwindow.cpp
        mainwindow.h
        models_inference_thread.cpp
        models_inference_thread.h
        param_assess_thread.cpp
        param_assess_thread.h
        param_assess_widget.cpp
        param_assess_widget.h
        param_assesser.cpp
        param_assesser.h
        param_display_item_widget.cpp
        param_display_item_widget.h
        param_display_widget.cpp
        param_display_widget.h
        param_premium_widget.cpp
        param_premium_widget.h
        progress_super_thread.cpp
        progress_super_thread.h
        quality_control_thread.cpp
        quality_control_thread.h
        quality_control_widget.cpp
        quality_control_widget.h
        quality_detail_play_thread.cpp
        quality_detail_play_thread.h
        quality_detail_widget.cpp
        quality_detail_widget.h
        quality_display_widget.cpp
        quality_display_widget.h
        spec_classification_inferer.cpp
        spec_classification_inferer.h
        type_conversion.cpp
        type_conversion.h
        type_define.h
        type_registering.h
        typemapping.h
        video_stream_thread.cpp
        video_stream_thread.h
        view_classification_inferer.cpp
        view_classification_inferer.h
        view_cls_inferer.cpp
        view_cls_inferer.h
        view_progress_widget.cpp
        view_progress_widget.h)

target_link_libraries(24_02_04_Quality_Param_Assess_GUI
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        ${OpenCV_LIBS}
        ${YAML_CPP_LIBRARY}
        ${TENSORRT_ROOT}/lib/nvcaffe_parser.lib 
        ${TENSORRT_ROOT}/lib/nvinfer.lib
        ${FFMPEG_ROOT}/lib/avcodec.lib
        ${FFMPEG_ROOT}/lib/avutil.lib
        ${FFMPEG_ROOT}/lib/swscale.lib
        ${FFMPEG_ROOT}/lib/avdevice.lib
        ${FFMPEG_ROOT}/lib/avfilter.lib
        ${FFMPEG_ROOT}/lib/avformat.lib
        ${FFMPEG_ROOT}/lib/swresample.lib

)

if (WIN32 AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(DEBUG_SUFFIX)
    if (MSVC AND CMAKE_BUILD_TYPE MATCHES "Debug")
        set(DEBUG_SUFFIX "d")
    endif ()
    set(QT_INSTALL_PATH "${CMAKE_PREFIX_PATH}")
    if (NOT EXISTS "${QT_INSTALL_PATH}/bin")
        set(QT_INSTALL_PATH "${QT_INSTALL_PATH}/..")
        if (NOT EXISTS "${QT_INSTALL_PATH}/bin")
            set(QT_INSTALL_PATH "${QT_INSTALL_PATH}/..")
        endif ()
    endif ()
    if (EXISTS "${QT_INSTALL_PATH}/plugins/platforms/qwindows${DEBUG_SUFFIX}.dll")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>/plugins/platforms/")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                "${QT_INSTALL_PATH}/plugins/platforms/qwindows${DEBUG_SUFFIX}.dll"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>/plugins/platforms/")
    endif ()
    foreach (QT_LIB Core Gui Widgets)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                "${QT_INSTALL_PATH}/bin/Qt5${QT_LIB}${DEBUG_SUFFIX}.dll"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
    endforeach (QT_LIB)
endif ()
if(OpenCV_FOUND)
    message(STATUS "OpenCV 库找到版本: ${OpenCV_VERSION}")
    message(STATUS "OpenCV 包含路径: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV 库文件: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "未找到 OpenCV 库")
endif()

if(YAML_CPP_LIBRARY)
    message(STATUS "Found yaml-cpp library: ${YAML_CPP_LIBRARY}")
else()
    message(FATAL_ERROR "yaml-cpp library not found in ${YAML_CPP_INSTALL_PATH}/lib")
endif()
