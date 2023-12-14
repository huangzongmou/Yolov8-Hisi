cmake_minimum_required(VERSION 3.14.1)

# project information
project(board)

# set path
set(TOP_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/../)
set(CMAKE_INSTALL_PREFIX ${TOP_DIR}/output/ CACHE STRING "install prefix" FORCE)

# set compile options
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-fstack-protector-strong -fPIC -fstack-check -fno-common)
add_compile_options(-D_FORTIFY_SOURCE=2 -g -O0 -ftrapv)
# add_compile_options(-Wall -Werror)
add_link_options(-Wl,-z,relro,-z,now -Wl,-z,noexecstack)

# set include path and link path
include_directories(
        /root/out/include
        /root/out/include/svp_npu
)

link_directories(
        /root/out/lib
        /root/out/lib/svp_npu
)

# make execute 
add_executable(main
        main.cpp 
        DetectPostprocess.cpp
        DetectCfgParse.cpp
        Utils.cpp
	ForwardEngine.cpp
        Yolo.cpp)

target_link_libraries(main pthread svp_acl protobuf-c securec ss_mpi ss_voice_engine ss_upvqe ss_dnvqe)

# install to output directory
install(TARGETS main RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})

