cmake_minimum_required(VERSION 3.20)
project(gesture_inference)

set(CMAKE_CXX_STANDARD 17)

set(ONNXRUNTIME_DIR "C:/libs/onnxruntime")

include_directories(${ONNXRUNTIME_DIR}/include)

add_executable(gesture_inference gesture_inference.cpp)

target_link_libraries(gesture_inference
    ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib
)

# Copy DLL next to the exe after build
add_custom_command(TARGET gesture_inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
    $<TARGET_FILE_DIR:gesture_inference>
)