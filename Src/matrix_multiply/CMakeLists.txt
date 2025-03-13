set(BLOCK_NAME cu_matrix_multiply)

include_directories(
  ${SOURCE_DIR}
  ${CUDA_TOOLKIT_INCLUDE}
  ${NVTX_INCLUDE_DIR}
  ${NVML_INCLUDE_DIR}
  ${FREEGLUT_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
  ${CUDA_TOOLKIT_SAMPLES_DIR}/common/inc
  ${PYTHON_INCLUDE_DIRS}
  ${PYBIND11_INCLUDE_DIRS}
)

#######################################
#######################################
file(GLOB HEADERS *.h *.cuh)
source_group("Headers" FILES ${HEADERS})
cuda_add_library(${BLOCK_NAME} MODULE
	mm.cu
	pybind11mm.cpp
OPTIONS --cudart=shared -DWITH_OPENGL -DOPTIMIZATION_STEP=0x0 -lineinfo -arch sm_52 -use_fast_math -maxrregcount 104
DEBUG -Xptxas -v -g -G 
)

set_target_properties (${BLOCK_NAME} PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX ".pyd"
)

target_link_libraries( 
${BLOCK_NAME}
${CUDA_CUDA_LIBRARY}
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
${NVML_DEBUG_LIBRARIES}
${NVTX_DEBUG_LIBRARIES}
${FREEGLUT_DEBUG_LIBRARIES}
${GLEW_DEBUG_LIBRARIES}
${PYTHON_LIBRARIES}
)

