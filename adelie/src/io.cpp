#include "decl.hpp"
#include <adelie_core/io/io_snp_unphased.hpp>
#include <adelie_core/io/io_snp_phased_ancestry.hpp>

namespace py = pybind11;
namespace ad = adelie_core;

void io_snp_base(py::module_& m)
{
    using io_t = ad::io::IOSNPBase;
    using string_t = typename io_t::string_t;
    py::class_<io_t>(m, "IOSNPBase")
        .def(py::init<string_t>(),
            py::arg("filename")
        )
        .def("endian", &io_t::endian)
        .def("read", &io_t::read)
        ;
}

void io_snp_unphased(py::module_& m)
{
    using io_t = ad::io::IOSNPUnphased;
    using base_t = typename io_t::base_t;
    using string_t = typename io_t::string_t;
    py::class_<io_t, base_t>(m, "IOSNPUnphased")
        .def(py::init<string_t>(),
            py::arg("filename")
        )
        .def("rows", &io_t::rows)
        .def("snps", &io_t::snps)
        .def("cols", &io_t::cols)
        .def("outer", &io_t::outer)
        .def("nnz", &io_t::nnz)
        .def("inner", &io_t::inner)
        .def("value", &io_t::value)
        .def("to_dense", &io_t::to_dense)
        .def("write", &io_t::write,
            py::arg("calldata").noconvert(),
            py::arg("n_threads")
        )
        ;
}

void io_snp_phased_ancestry(py::module_& m)
{
    using io_t = ad::io::IOSNPPhasedAncestry;
    using base_t = typename io_t::base_t;
    using string_t = typename io_t::string_t;
    py::class_<io_t, base_t>(m, "IOSNPPhasedAncestry")
        .def(py::init<string_t>(),
            py::arg("filename")
        )
        .def("rows", &io_t::rows)
        .def("snps", &io_t::snps)
        .def("ancestries", &io_t::ancestries)
        .def("cols", &io_t::cols)
        .def("outer", &io_t::outer)
        .def("nnz", &io_t::nnz)
        .def("inner", &io_t::inner)
        .def("ancestry", &io_t::ancestry)
        .def("to_dense", &io_t::to_dense)
        .def("write", &io_t::write,
            py::arg("calldata").noconvert(),
            py::arg("ancestries").noconvert(),
            py::arg("A"),
            py::arg("n_threads")
        )
        ;
}

void register_io(py::module_& m)
{
    io_snp_base(m);
    io_snp_unphased(m);
    io_snp_phased_ancestry(m);
}