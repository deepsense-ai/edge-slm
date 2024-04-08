#!/usr/bin/env bash

create_package() {
    pushd $1
    if [ -n "$2" ]; then
    conan create . --build=missing --version=$2
    conan create . --profile=android --build=missing --version=$2
    else
    conan create . --build=missing
    conan create . --profile=android --build=missing
    fi
    popd
}


pushd recipes

create_package OpenBLAS/all
create_package faiss/all
create_package llamacpp

popd
