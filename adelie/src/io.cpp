#include "decl.hpp"
#include <adelie_core/io/io_snp_unphased.hpp>
#include <adelie_core/io/io_snp_phased_ancestry.hpp>

namespace py = pybind11;
namespace ad = adelie_core;

void io_snp_base(py::module_& m)
{
    using io_t = ad::io::IOSNPBase<>;
    using string_t = typename io_t::string_t;
    py::class_<io_t>(m, "IOSNPBase")
        .def(py::init<
            const string_t&,
            const string_t&
        >(),
            py::arg("filename"),
            py::arg("read_mode")
        )
        .def_property_readonly("endian", [](const io_t& io) -> string_t { 
            return io.endian() ? "big" : "little";
        }, R"delimiter(
        Endianness used in the file.
        It is ``"big"`` if the system is big-endian otherwise ``"little"``.

        .. note::
            We recommend that users read/write from/to the file on the *same* machine.
            The ``.snpdat`` format depends on the endianness of the machine.
            So, unless the endianness is the same across two different machines,
            it is undefined behavior reading a file that was generated on a different machine.

        )delimiter")
        .def("read", &io_t::read, R"delimiter(
        Reads and loads the matrix from file.

        Returns
        -------
        total_bytes : int
            Number of bytes read.
        )delimiter")
        ;
}

void io_snp_unphased(py::module_& m)
{
    using io_t = ad::io::IOSNPUnphased<>;
    using base_t = typename io_t::base_t;
    using string_t = typename io_t::string_t;
    using vec_impute_t = typename io_t::vec_impute_t;
    using colarr_value_t = typename io_t::colarr_value_t;
    py::class_<io_t, base_t>(m, "IOSNPUnphased")
        .def(py::init<
            const string_t&,
            const string_t&
        >(),
            py::arg("filename"),
            py::arg("read_mode")
        )
        .def_property_readonly("rows", &io_t::rows, "Number of rows.")
        .def_property_readonly("snps", &io_t::snps, "Number of SNPs.")
        .def_property_readonly("cols", &io_t::cols, "Number of columns.")
        .def_property_readonly("nnz", &io_t::nnz, "Number of non-zero entries for each column.")
        .def_property_readonly("nnm", &io_t::nnm, R"delimiter(
        Number of non-missing entries for each column.

        .. note::
            Missing values are counted even if you wrote the matrix
            with imputation method as ``"zero"``.

        )delimiter")
        .def_property_readonly("impute", &io_t::impute, "Imputed value for each column.")
        .def("to_dense", &io_t::to_dense, 
            py::arg("n_threads")=1,
        R"delimiter(
        Creates a dense SNP unphased matrix from the file.

        .. note::
            The missing values are *always* encoded as ``-9``
            even if they were different (negative) values when writing to the file.

        Parameters
        ----------
        n_threads : int, optional
            Number of threads.
            Default is ``1``.

        Returns
        -------
        dense : (n, p) np.ndarray
            Dense SNP unphased matrix.
        )delimiter")
        .def("write", [](
            const io_t& io, 
            const Eigen::Ref<const colarr_value_t>& calldata,
            const std::string& impute_method_str,
            Eigen::Ref<vec_impute_t> impute,
            size_t n_threads
        ) {
            std::tuple<size_t, std::unordered_map<std::string, double>> out;
            std::string error;
            try {
                out = io.write(calldata, impute_method_str, impute, n_threads);
            } catch (const std::exception& e) {
                error = e.what();
            }
            return std::make_tuple(
                std::get<0>(out),
                std::get<1>(out),
                error
            );
        },
            py::arg("calldata").noconvert(),
            py::arg("impute_method"),
            py::arg("impute").noconvert(),
            py::arg("n_threads")
        )
        ;
}

void io_snp_phased_ancestry(py::module_& m)
{
    using io_t = ad::io::IOSNPPhasedAncestry<>;
    using base_t = typename io_t::base_t;
    using string_t = typename io_t::string_t;
    using colarr_value_t = typename io_t::colarr_value_t;
    py::class_<io_t, base_t>(m, "IOSNPPhasedAncestry")
        .def(py::init<
            const string_t&,
            const string_t&
        >(),
            py::arg("filename"),
            py::arg("read_mode")
        )
        .def_property_readonly("rows", &io_t::rows, "Number of rows.")
        .def_property_readonly("snps", &io_t::snps, "Number of SNPs.")
        .def_property_readonly("cols", &io_t::cols, "Number of columns.")
        .def_property_readonly("ancestries", &io_t::ancestries, "Number of ancestries.")
        .def_property_readonly("nnz", &io_t::nnz, "Number of non-zero entries for each column.")
        .def("to_dense", &io_t::to_dense, 
            py::arg("n_threads")=1,
        R"delimiter(
        Creates a dense SNP phased, ancestry matrix from the file.

        Parameters
        ----------
        n_threads : int, optional
            Number of threads.
            Default is ``1``.

        Returns
        -------
        dense : (n, s*A) np.ndarray
            Dense SNP phased, ancestry matrix.
        )delimiter")
        .def("write", [](
            const io_t& io,
            const Eigen::Ref<const colarr_value_t>& calldata,
            const Eigen::Ref<const colarr_value_t>& ancestries,
            size_t A,
            size_t n_threads
        ) {
            std::tuple<size_t, std::unordered_map<std::string, double>> out;
            std::string error;
            try {
                out = io.write(calldata, ancestries, A, n_threads);
            } catch (const std::exception& e) {
                error = e.what();
            }
            return std::make_tuple(
                std::get<0>(out),
                std::get<1>(out),
                error
            );
        },
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