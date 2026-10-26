#pragma once
#include <adelie_core/io/io_snp_base.hpp>
#include <adelie_core/io/io_snp_phased_ancestry.hpp>
#include <adelie_core/io/io_snp_unphased.hpp>
#include <adelie_core/io/io_snp_combine_r.hpp>
#include <adelie_core/io/io_snp_combine_s.hpp>

extern template class adelie_core::io::IOSNPBase<>;
extern template class adelie_core::io::IOSNPPhasedAncestry<>;
extern template class adelie_core::io::IOSNPUnphased<>;
extern template class adelie_core::io::IOSNPCombineR<>;
extern template class adelie_core::io::IOSNPCombineS<>;