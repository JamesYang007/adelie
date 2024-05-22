#pragma once
#include <string>
#include <vector>
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/io/io_snp_phased_ancestry.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType,
          class MmapPtrType=std::unique_ptr<char, std::function<void(char*)>>>
class MatrixNaiveSNPPhasedAncestry: public MatrixNaiveBase<ValueType>
{
public:
    using base_t = MatrixNaiveBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    using string_t = std::string;
    using io_t = io::IOSNPPhasedAncestry<MmapPtrType>;
    
protected:
    const io_t _io;             // IO handler
    const size_t _n_threads;    // number of threads
    util::rowvec_type<char> _bbuff;
    vec_index_t _ibuff;
    vec_value_t _buff;

    static auto init_io(
        const string_t& filename,
        const string_t& read_mode
    )
    {
        io_t io(filename, read_mode);
        io.read();
        return io;
    }

    ADELIE_CORE_STRONG_INLINE
    value_t _cmul(
        int j,
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights,
        size_t n_threads
    ) 
    {
        return snp_phased_ancestry_dot(
            _io, j, v * weights, n_threads, _buff
        );
    }

    ADELIE_CORE_STRONG_INLINE
    void _ctmul(
        int j,
        value_t v,
        Eigen::Ref<vec_value_t> out,
        size_t n_threads
    )
    {
        return snp_phased_ancestry_axi(
            _io, j, v, out, n_threads
        );
    }

    auto ancestries() const { return _io.ancestries(); }

public:
    explicit MatrixNaiveSNPPhasedAncestry(
        const string_t& filename,
        const string_t& read_mode,
        size_t n_threads
    ): 
        _io(init_io(filename, read_mode)),
        _n_threads(n_threads),
        _bbuff(_io.rows()),
        _ibuff(_io.rows()),
        _buff(n_threads * _io.ancestries())
    {
        if (n_threads < 1) {
            throw util::adelie_core_error("n_threads must be >= 1.");
        }
        _bbuff.setZero();
    }

    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
        return _cmul(j, v, weights, _n_threads);
    }

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_ctmul(j, out.size(), rows(), cols());
        _ctmul(j, v, out, _n_threads);
    }

    void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
        if (_buff.size() < q * _n_threads) _buff.resize(q * _n_threads);
        snp_phased_ancestry_block_dot(
            _io, j, q, v * weights, out, _n_threads, _buff
        );
    }

    void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
        if (_buff.size() < q * _n_threads) _buff.resize(q * _n_threads);
        snp_phased_ancestry_block_axi(
            _io, j, q, v, out, _n_threads
        );
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        const auto routine = [&](int t) {
            out[t] = _cmul(t, v, weights, 1);
        };
        if (_n_threads <= 1) {
            for (int t = 0; t < cols(); ++t) routine(t);
        } else {
            #pragma omp parallel for schedule(static) num_threads(_n_threads)
            for (int t = 0; t < cols(); ++t) routine(t);
        }
    }

    void cov(
        int j, int q,
        const Eigen::Ref<const vec_value_t>& sqrt_weights,
        Eigen::Ref<colmat_value_t> out,
        Eigen::Ref<colmat_value_t> buffer
    ) override
    {
        base_t::check_cov(
            j, q, sqrt_weights.size(), 
            out.rows(), out.cols(), buffer.rows(), buffer.cols(), 
            rows(), cols()
        );
        
        const auto A = ancestries();

        out.setZero(); // don't parallelize! q is usually small

        int n_solved0 = 0;
        while (n_solved0 < q) {
            const auto begin0 = j + n_solved0;
            const auto snp0 = begin0 / A;
            const auto ancestry_lower0 = begin0 % A;
            const auto ancestry_upper0 = std::min<int>(ancestry_lower0 + q - n_solved0, A);

            int n_solved1 = 0;
            while (n_solved1 <= n_solved0) {
                const auto begin1 = j + n_solved1;
                const auto snp1 = begin1 / A;
                const auto ancestry_lower1 = begin1 % A;
                const auto ancestry_upper1 = std::min<int>(ancestry_lower1 + q - n_solved1, A);

                if (n_solved0 == n_solved1) {
                    const auto begin = begin0;
                    const auto snp = snp0;
                    const auto a_low = ancestry_lower0;
                    const auto a_high = ancestry_upper0;
                    const auto a_size = a_high - a_low;

                    // increase buffer including cross-term computation part as well
                    if (_buff.size() < a_size * a_size * _n_threads) {
                        _buff.resize(a_size * a_size * _n_threads);
                    }

                    // compute quadratic diagonal part
                    auto out_diag = out.diagonal().segment(n_solved0, a_size);
                    snp_phased_ancestry_block_dot(
                        _io, begin0, a_size, sqrt_weights.square(), out_diag, _n_threads, _buff
                    );

                    // compute cross-terms
                    //for (int k0 = 0; k0 < a_size; ++k0) {
                    //    // cache hap0 information
                    //    size_t nnz = 0;
                    //    auto it0 = _io.begin(snp, a_low + k0, 0);
                    //    const auto end0 = _io.end(snp, a_low + k0, 0);
                    //    for (; it0 != end0; ++it0) {
                    //        const auto idx = *it0;
                    //        _bbuff[idx] = true;
                    //        _ibuff[nnz] = idx;
                    //        ++nnz;
                    //    }

                    //    // loop through hap1's ancestries
                    //    for (int k1 = 0; k1 < a_size; ++k1) {
                    //        auto it1 = _io.begin(snp, a_low + k1, 1);
                    //        const auto end1 = _io.end(snp, a_low + k1, 1);
                    //        value_t sum = 0;
                    //        for (; it1 != end1; ++it1) {
                    //            const auto idx = *it1;
                    //            const auto sqrt_w = sqrt_weights[idx];
                    //            sum += sqrt_w * sqrt_w * _bbuff[idx];
                    //        }

                    //        const auto kk0 = n_solved0 + k0;
                    //        const auto kk1 = n_solved0 + k1;
                    //        out(kk0, kk1) += sum;
                    //        out(kk1, kk0) += sum;
                    //    }

                    //    // keep invariance by populating with false
                    //    for (size_t i = 0; i < nnz; ++i) {
                    //        _bbuff[_ibuff[i]] = false;
                    //    }
                    //}

                    #pragma omp parallel for schedule(static) num_threads(_n_threads) collapse(2) if(_n_threads > 1)
                    for (int k0 = 0; k0 < a_size; ++k0) {
                        for (int k1 = 0; k1 < a_size; ++k1) {
                            auto it0 = _io.begin(snp, a_low + k0, 0);
                            const auto end0 = _io.end(snp, a_low + k0, 0);
                            auto it1 = _io.begin(snp, a_low + k1, 1);
                            const auto end1 = _io.end(snp, a_low + k1, 1);

                            value_t sum = 0;
                            while (
                                (it0 != end0) &&
                                (it1 != end1)
                            ) {
                                const auto idx0 = *it0;
                                const auto idx1 = *it1;
                                if (idx0 < idx1) {
                                    ++it0; 
                                    continue;
                                }
                                else if (idx0 > idx1) {
                                    ++it1;
                                    continue;
                                } 
                                else {
                                    const auto sqrt_w = sqrt_weights[idx0];
                                    sum += sqrt_w * sqrt_w;
                                    ++it0;
                                    ++it1;
                                }
                            }

                            const auto kk0 = n_solved0 + k0;
                            const auto kk1 = n_solved0 + k1;
                            out(kk0, kk1) += sum * (1 + (kk0 == kk1));
                        }
                    }

                    for (int k0 = 0; k0 < a_size; ++k0) {
                        for (int k1 = 0; k1 < k0; ++k1) {
                            const auto kk0 = n_solved0 + k0;
                            const auto kk1 = n_solved0 + k1;
                            const auto tmp = out(kk0, kk1);
                            out(kk0, kk1) += out(kk1, kk0);
                            out(kk1, kk0) += tmp;
                        }
                    }

                    n_solved1 += ancestry_upper1 - ancestry_lower1;
                    continue;
                }

                /* general routine */

                const auto ancestry_size0 = ancestry_upper0-ancestry_lower0;
                const auto ancestry_size1 = ancestry_upper1-ancestry_lower1;
                for (int a0 = 0; a0 < ancestry_size0; ++a0) {
                    size_t nnz = 0;
                    for (int hap0 = 0; hap0 < io_t::n_haps; ++hap0) {
                        auto it = _io.begin(snp0, ancestry_lower0+a0, hap0);
                        const auto end = _io.end(snp0, ancestry_lower0+a0, hap0);
                        for (; it != end; ++it) {
                            const auto idx = *it;
                            if (!_bbuff[idx]) {
                                _ibuff[nnz] = idx;
                                ++nnz;
                            }
                            _bbuff[idx] += 1;
                        }
                    }

                    for (int a1 = 0; a1 < ancestry_size1; ++a1) {
                        const auto sum = snp_phased_ancestry_dot(
                            _io, begin1 + a1, 
                            vec_value_t::NullaryExpr(sqrt_weights.size(), [&](auto i) {
                                const auto sqrt_wi = sqrt_weights[i];
                                return sqrt_wi * sqrt_wi * _bbuff[i];
                            }),
                            _n_threads,
                            _buff
                        );
                        const auto kk0 = n_solved0 + a0;
                        const auto kk1 = n_solved1 + a1;
                        out(kk0, kk1) = sum;
                        out(kk1, kk0) = sum;
                    }

                    for (size_t i = 0; i < nnz; ++i) {
                        _bbuff[_ibuff[i]] = 0;
                    }
                }

                n_solved1 += ancestry_upper1 - ancestry_lower1;
            }
            n_solved0 += ancestry_upper0 - ancestry_lower0;
        }     
    }

    int rows() const override { return _io.rows(); }
    int cols() const override { return _io.snps() * ancestries(); }

    void sp_btmul(
        const sp_mat_value_t& v,
        Eigen::Ref<rowmat_value_t> out
    ) override
    {
        base_t::check_sp_btmul(
            v.rows(), v.cols(), out.rows(), out.cols(), rows(), cols()
        );
        const auto A = ancestries();
        const auto routine = [&](int k) {
            typename sp_mat_value_t::InnerIterator it(v, k);
            auto out_k = out.row(k);
            out_k.setZero();
            for (; it; ++it) {
                _ctmul(it.index(), it.value(), out_k, 1);
            }
        };
        if (_n_threads <= 1) {
            for (int k = 0; k < v.outerSize(); ++k) routine(k);
        } else {
            #pragma omp parallel for schedule(static) num_threads(_n_threads)
            for (int k = 0; k < v.outerSize(); ++k) routine(k);
        }
    }
};

} // namespace matrix
} // namespace adelie_core