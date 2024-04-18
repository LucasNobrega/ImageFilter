set(LIBRARY_SOURCES
    "Abstract/AbstractFilter.cpp"
    "Abstract/AbstractFilterCUDA.cu"
    "Box/BoxFilter.cpp"
    "Box/BoxFilterCUDA.cu"
    "Laplacian/LaplacianFilter.cpp"
    "Laplacian/LaplacianFilterCUDA.cu"
    "Median/MedianBlurFilter.cpp"
    "Median/MedianBlurFilterCUDA.cu"
    "Sharpening/SharpeningFilter.cpp"
    "Sharpening/SharpeningFilterCUDA.cu"
    "SobelEdgeDetection/SobelEdgeDetectionFilter.cpp"
    "SobelEdgeDetection/SobelEdgeDetectionFilterCUDA.cu"
    "TotalVariation/TotalVariationFilter.cpp"
    "TotalVariation/TotalVariationFilterCUDA.cu")
set(LIBRARY_HEADERS
    "Abstract/AbstractFilter.hpp"
    "Abstract/CUDARunTimeConfig.hpp"
    "Abstract/AbstractFilterCUDA.hpp"
    "Box/BoxFilter.hpp"
    "Box/BoxFilterCUDA.hpp"
    "Laplacian/LaplacianFilter.hpp"
    "Laplacian/LaplacianFilterCUDA.hpp"
    "Median/MedianBlurFilter.hpp"
    "Median/MedianBlurFilterCUDA.hpp"
    "Sharpening/SharpeningFilter.hpp"
    "Sharpening/SharpeningFilterCUDA.hpp"
    "SobelEdgeDetection/SobelEdgeDetectionFilter.hpp"
    "SobelEdgeDetection/SobelEdgeDetectionFilterCUDA.hpp"
    "TotalVariation/TotalVariationFilter.hpp"
    "TotalVariation/TotalVariationFilterCUDA.hpp")

set(LIBRARY_INCLUDES
    "./"
    "./Abstract"
    "./Box"
    "./Laplacian"
    "./Median"
    "./Sharpening"
    "./SobelEdgeDetection"
    "./test"
    "./TotalVariation")

cuda_add_library(
    ${LIB_IMAGE_FILTER_NAME}
    STATIC
    ${LIBRARY_SOURCES}
    ${LIBRARY_HEADERS})
target_include_directories(${LIB_IMAGE_FILTER_NAME} PUBLIC ${LIBRARY_INCLUDES})
target_link_libraries(imagefilter ${OpenCV_LIBS} ${CUDA_LIBRARIES})

if(${ENABLE_WARNINGS})
    target_set_warnings(
        TARGET
        ${LIB_IMAGE_FILTER_NAME}
        ENABLE
        ${ENABLE_WARNINGS}
        AS_ERRORS
        ${ENABLE_WARNINGS_AS_ERRORS})
endif()

if(${ENABLE_LTO})
    target_enable_lto(
        TARGET
        ${LIB_IMAGE_FILTER_NAME}
        ENABLE
        ON)
endif()

if(${ENABLE_CLANG_TIDY})
    add_clang_tidy_to_target(${LIB_IMAGE_FILTER_NAME})
endif()

#add_subdirectory(test)
