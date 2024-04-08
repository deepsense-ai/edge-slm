#!/usr/bin/env python3

from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps
from conan.tools.scm import Git
from conan.tools.files import get, copy, mkdir, rename
import os

class LlamaCppConan(ConanFile):
    name = "llamacpp"
    package_type = "library"
    version = "3a0345970ed0353fa857df3c8a62a2b3318b1364"
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

    def source(self):
        git = Git(self)
        git.clone("https://github.com/ggerganov/llama.cpp.git", ".")
        git.run("submodule update --init --recursive")
        git.checkout(commit=self.version)

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        pass

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.generate()

    def build(self):
        cmake = CMake(self)
        variables={
            "LLAMA_BUILD_TESTS": "OFF",
            "LLAMA_BUILD_EXAMPLES": "ON",
            "LLAMA_BUILD_SERVER": "OFF",
        }

        if self.settings.os == "Android":
            variables.update(
                {
                    "CMAKE_C_FLAGS": "-march=armv8.4a+dotprod",
                    "CMAKE_CXX_FLAGS": "-march=armv8.4a+dotprod",
                }
            )

        cmake.configure(variables=variables)
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

        common_dir = os.path.join(self.source_folder, "common")
        build_common_dir = os.path.join(self.build_folder, "common")
        include_dir = os.path.join(self.package_folder, "include")
        llamacpp_include_dir = os.path.join(include_dir, "llamacpp")
        mkdir(self, llamacpp_include_dir)

        headers = [os.path.join(include_dir, f) for f in os.listdir(include_dir)
                   if os.path.splitext(f)[1] == '.h']

        for header_path in headers:
            filename = os.path.basename(header_path)
            rename(self, header_path, os.path.join(llamacpp_include_dir, filename))

        lib_dir = os.path.join(self.package_folder, "lib")

        copy(self, "*.h", common_dir, llamacpp_include_dir, keep_path=False)
        copy(self, "*.a", build_common_dir, lib_dir, keep_path=False)

    def package_info(self):
        self.cpp_info.libs = ["llama", "common"]
