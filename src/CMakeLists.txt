option(ENABLE_CULLING "Set to true to cull back-facing triangles" ON)

set(COMMON_SRCS
    bvh.h
    intr_bvh.c
    intr.h
    linalg.inl
    mem.c
    mem.h
    shader.h
    view.h)

add_library(dream_generic
    ${COMMON_SRCS}
    generic/bvh.c
    generic/intr_box.c
    generic/intr_frustum.c
    generic/intr_tri.c
    generic/shader.c
    generic/view.c)

add_library(dream_x86
    ${COMMON_SRCS}
    x86/bvh.c
    x86/intr_box.c
    x86/intr_frustum.c
    x86/intr_tri.c
    x86/shader.c
    x86/shader_utils.inl
    x86/view.c
    x86/x86.inl)

if (ENABLE_CULLING)
    target_compile_options(dream_generic PRIVATE -DCULLING)
    target_compile_options(dream_x86 PRIVATE -DCULLING)
endif ()

include(CheckLibraryExists)
check_library_exists(m sin "" HAS_MATH_LIB)
set(MATH_LIB "")
if (HAS_MATH_LIB)
    set(MATH_LIB "-lm")
endif ()

include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT NO_IPO_REASON)
if (IPO_SUPPORTED)
    set_property(TARGET dream_generic dream_x86 PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else ()
    message(STATUS "IPO is OFF: ${IPO_REASON}")
endif ()

target_include_directories(dream_generic INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(dream_x86 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dream_generic PUBLIC ${MATH_LIB})
target_link_libraries(dream_x86     PUBLIC ${MATH_LIB})
