#include "decl.hpp"
#include <adelie_core/io/io_snp_unphased.hpp>

namespace py = pybind11;
namespace ad = adelie_core;

void io_snp_unphased(py::module_& m)
{
    using io_t = ad::io::IOSNPUnphased;
    py::class_<io_t>(m, "IOSNPUnphased")
        .def(py::init<std::string>(),
            py::arg("filename")
        )
        .def("endian", &io_t::endian)
        .def("rows", &io_t::rows)
        .def("cols", &io_t::cols)
        .def("outer", &io_t::outer)
        .def("nnz", &io_t::nnz)
        .def("inner", &io_t::inner)
        .def("value", &io_t::value)
        .def("to_dense", &io_t::to_dense)
        .def("read", &io_t::read)
        .def("write", &io_t::write,
            py::arg("calldata").noconvert(),
            py::arg("n_threads")
        )
        ;
}

void register_io(py::module_& m)
{
    io_snp_unphased(m);
}