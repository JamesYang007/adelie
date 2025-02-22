#pragma once
#include <adelie_core/io/io_snp_base.hpp>
#include <adelie_core/io/io_snp_phased_ancestry.hpp>
#include <adelie_core/io/io_snp_unphased.hpp>

extern template class adelie_core::io::IOSNPBase<>;
extern template class adelie_core::io::IOSNPPhasedAncestry<>;
extern template class adelie_core::io::IOSNPUnphased<>;