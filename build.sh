#!/bin/bash
set -e

build_dir=build/
output_dir=output/
if [ -d "${build_dir}" ]; then
  rm -fr "${build_dir:?}"/*
else
  mkdir -p "${build_dir}"
fi
if [ -d "${output_dir}" ]; then
  rm -fr "${output_dir:?}"/*
else
  mkdir -p "${output_dir}"
fi

pushd "${build_dir}" || exit
cmake ../code -DCMAKE_CXX_COMPILER=aarch64-mix410-linux-g++ -DCMAKE_SKIP_RPATH=TRUE

cpu_processor_num=$(grep processor /proc/cpuinfo | wc -l)
job_num=$(expr "$cpu_processor_num" \* 2)
echo Parallel job num is "$job_num"
make -j"$job_num"
make install -j"$job_num"
popd || exit

