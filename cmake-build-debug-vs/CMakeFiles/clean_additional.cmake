# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Debug")
  file(REMOVE_RECURSE
  "24_02_04_Quality_Param_Assess_GUI_autogen"
  "CMakeFiles\\24_02_04_Quality_Param_Assess_GUI_autogen.dir\\AutogenUsed.txt"
  "CMakeFiles\\24_02_04_Quality_Param_Assess_GUI_autogen.dir\\ParseCache.txt"
  )
endif()
