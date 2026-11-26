# Tests of the helpers that the plugins share and that need no device.
add_unittest(OffloadUnitTests "PluginUtils.unittests"
  PTXTest.cpp)
target_include_directories("PluginUtils.unittests" PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../plugins-nextgen/common/include)
target_link_libraries("PluginUtils.unittests" PRIVATE LLVMSupport)
