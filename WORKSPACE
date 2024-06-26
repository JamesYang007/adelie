load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Google benchmark
http_archive(
    name = "com_github_google_benchmark",
    url = "https://github.com/google/benchmark/archive/refs/tags/v1.7.1.zip",
    sha256 = "aeec52381284ec3752505a220d36096954c869da4573c2e1df3642d2f0a4aac6",
    strip_prefix = "benchmark-1.7.1",
)

# Rules CC
http_archive(
    name = "rules_cc",
    sha256 = "9a446e9dd9c1bb180c86977a8dc1e9e659550ae732ae58bd2e8fd51e15b2c91d",
    strip_prefix = "rules_cc-262ebec3c2296296526740db4aefce68c80de7fa",
    urls = ["https://github.com/bazelbuild/rules_cc/archive/262ebec3c2296296526740db4aefce68c80de7fa.zip"],
)

# Add OMP 
new_local_repository(
    name = "omp",
    path = "/opt/homebrew/opt/libomp/include",
    build_file_content = """
package(default_visibility = ["//visibility:public"])
cc_library(
    name = "headers",
    hdrs = glob(["**/*.h"])
)
"""
)

EIGEN_VERSION = "3.4.0"

http_archive(
    name = "eigen",
    build_file_content =
        """
# TODO(keir): Replace this with a better version, like from TensorFlow.
# See https://github.com/ceres-solver/ceres-solver/issues/337.
cc_library(
    name = 'eigen',
    srcs = [],
    includes = ['.'],
    hdrs = glob(['Eigen/**', 'unsupported/Eigen/**']),
    visibility = ['//visibility:public'],
)
""",
    sha256 = "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72",
    strip_prefix = "eigen-{}".format(EIGEN_VERSION),
    urls = ["https://gitlab.com/libeigen/eigen/-/archive/{0}/eigen-{0}.tar.gz".format(EIGEN_VERSION)],
)