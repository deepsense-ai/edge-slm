#!/usr/bin/env python3

from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps
from conan.tools.scm import Git
from conan.tools.files import get

class FaissConan(ConanFile):
    name = "faiss"
    package_type = "library"
    version = "1.7.4"
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

    def source(self):
        get(self,
            **self.conan_data["sources"][self.version],
            strip_root=True,
            destination=self.source_folder
        )

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        self.requires("openblas/0.3.26")

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure(variables={
            "FAISS_ENABLE_GPU":"OFF",
            "FAISS_ENABLE_PYTHON":"OFF",
            "FAISS_ENABLE_RAFT":"OFF",
            "FAISS_ENABLE_C_API":"ON",
            "BUILD_TESTING": "OFF"
        })
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["faiss"]
