from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps


class RagRecipe(ConanFile):
    name = "rag"
    version = "1.0"
    package_type = "library"

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "CMakeLists.txt", "src/*", "include/*", "config/*", "tests/*", "benchmarks/*", "apps/*"

    def requirements(self):
        self.requires("fmt/10.2.1")
        self.requires("spdlog/1.13.0")
        self.requires("benchmark/1.8.3")
        self.requires("gtest/1.14.0")
        self.requires("faiss/1.7.4")
        self.requires("llamacpp/3a0345970ed0353fa857df3c8a62a2b3318b1364")
        self.requires("nlohmann_json/3.11.3")
        self.requires("boost/1.84.0")
        self.requires("tabulate/1.5")
        self.requires("kainjow-mustache/4.1")
        self.requires("magic_enum/0.9.5")

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")
        self.options["boost/*"].without_atomic = True
        self.options["boost/*"].without_chrono = True
        self.options["boost/*"].without_cobalt = True
        self.options["boost/*"].without_container = True
        self.options["boost/*"].without_context = True
        self.options["boost/*"].without_contract = True
        self.options["boost/*"].without_coroutine = True
        self.options["boost/*"].without_date_time = True
        self.options["boost/*"].without_exception = True
        self.options["boost/*"].without_fiber = True
        self.options["boost/*"].without_filesystem = True
        self.options["boost/*"].without_graph = True
        self.options["boost/*"].without_graph_parallel = True
        self.options["boost/*"].without_headers = True
        self.options["boost/*"].without_iostreams = True
        self.options["boost/*"].without_json = True
        self.options["boost/*"].without_locale = True
        self.options["boost/*"].without_log = True
        self.options["boost/*"].without_math = True
        self.options["boost/*"].without_mpi = True
        self.options["boost/*"].without_nowide = True
        self.options["boost/*"].without_program_options = False
        self.options["boost/*"].without_python = True
        self.options["boost/*"].without_random = True
        self.options["boost/*"].without_regex = True
        self.options["boost/*"].without_serialization = True
        self.options["boost/*"].without_stacktrace = True
        self.options["boost/*"].without_system = True
        self.options["boost/*"].without_test = True
        self.options["boost/*"].without_thread = True
        self.options["boost/*"].without_timer = True
        self.options["boost/*"].without_type_erasure = True
        self.options["boost/*"].without_url = True
        self.options["boost/*"].without_wave = True
        self.options["boost/*"].zlib = False
        self.options["boost/*"].bzip2 = False

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure(variables={
            "CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
        })
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["rag"]
