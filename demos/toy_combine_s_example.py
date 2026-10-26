import numpy as np
import adelie.io as io
import adelie.data as data
import os
import time

# This example demonstrates how to create a .snpdat file for 
# SNP both-ancestry data, write it to a file, read it back,
# and verify the contents, including a selected-ancestries mode.

np.random.seed(0)

# Toy data parameters
n_samples = 5
n_snps = 4
n_ancestries = 3

print("Generating toy phased data with parameters:")
print(f"  - Samples: {n_samples}")
print(f"  - SNPs: {n_snps}")
print(f"  - Ancestries: {n_ancestries}")

# Generate toy phased calldata (0/1 per hap) and phased ancestries
D = data.snp_phased_ancestry(n=n_samples, s=n_snps, A=n_ancestries, seed=0)
calldata = D["X"]               # shape (n, 2*s), entries 0/1 (per hap)
ancestries = D["ancestries"]    # shape (n, 2*s), entries in [0, A)

print(f"\nPhased calldata (shape: {calldata.shape}):")
print(calldata)

print(f"\nPhased ancestries (shape: {ancestries.shape}):")
print(ancestries)

# Build expected dense matrix for the both-ancestry format
# Per SNP j, block of size (2*A):
# - First A cols: mutated hap counts per ancestry a
# - Next  A cols: ancestry dosage counts per ancestry a

print("\nComputing expected both-ancestry dense matrix (full A)...")
start = time.time()
expected_dense = np.zeros((n_samples, n_snps * (2 * n_ancestries)), dtype=np.int8)

# Convenience views per hap
cal0 = calldata[:, ::2]
cal1 = calldata[:, 1::2]
anc0 = ancestries[:, ::2]
anc1 = ancestries[:, 1::2]

for a in range(n_ancestries):
    # mutated hap counts for ancestry a
    mut_counts = ((cal0 == 1) & (anc0 == a)).astype(np.int8) + ((cal1 == 1) & (anc1 == a)).astype(np.int8)
    expected_dense[:, a::(2 * n_ancestries)] = mut_counts
    # ancestry dosage counts for ancestry a
    dosage_counts = (anc0 == a).astype(np.int8) + (anc1 == a).astype(np.int8)
    expected_dense[:, (n_ancestries + a)::(2 * n_ancestries)] = dosage_counts

compute_time = time.time() - start
print(f"Done. Time: {compute_time:.6f}s")
print(f"Expected dense (full) shape: {expected_dense.shape}")
print(expected_dense)

# Write using io.snp_combine_s
output_filename = "combine_s.snpdat"
handler = io.snp_combine_s(filename=output_filename)

print(f"\nWriting both-ancestry data to '{output_filename}'...")
start = time.time()
total_bytes, benchmark = handler.write(
    calldata=calldata,
    ancestries=ancestries,
    A=n_ancestries,
    n_threads=2,
)
write_time = time.time() - start
print(f"Successfully wrote {total_bytes} bytes in {write_time:.6f}s.")

# Read back and verify
handler_read = io.snp_combine_s(filename=output_filename, read_mode="mmap")
handler_read.read()

print("Verifying metadata (full A)...")
assert handler_read.rows == n_samples
assert handler_read.snps == n_snps
assert handler_read.ancestries == n_ancestries
assert handler_read.cols == n_snps * (2 * n_ancestries)
print("  - OK")

print("Reconstructing dense matrix from file (full A)...")
dense_from_file = handler_read.to_dense(n_threads=2)
np.testing.assert_array_equal(dense_from_file, expected_dense)
print("  - Reconstruction matches expected.")

# ------------------------------------------------------------------
# Selected ancestries demo (e.g., only ancestries 0 and 2)
# ------------------------------------------------------------------
selected = np.array([0, 2], dtype=np.uint32)
output_filename_sel = "combine_s_selected.snpdat"
handler_sel = io.snp_combine_s(filename=output_filename_sel)

print(f"\nWriting both-ancestry data with selected ancestries {selected.tolist()} to '{output_filename_sel}'...")
start = time.time()
total_bytes_sel, benchmark_sel = handler_sel.write(
    calldata=calldata,
    ancestries=ancestries,
    A=n_ancestries,
    n_threads=2,
    selected_ancestries=selected,
)
print(f"Successfully wrote {total_bytes_sel} bytes in {time.time() - start:.6f}s.")

handler_sel_read = io.snp_combine_s(filename=output_filename_sel, read_mode="mmap")
handler_sel_read.read()

print("Verifying metadata (selected ancestries)...")
assert handler_sel_read.rows == n_samples
assert handler_sel_read.snps == n_snps
assert handler_sel_read.ancestries == len(selected)
assert handler_sel_read.cols == n_snps * (2 * len(selected))
print("  - OK")

print("Computing expected dense for selected ancestries...")
expected_dense_sel = np.zeros((n_samples, n_snps * (2 * len(selected))), dtype=np.int8)
for ai, a in enumerate(selected):
    # mutated counts for selected ancestry a
    mut_counts = ((cal0 == 1) & (anc0 == a)).astype(np.int8) + ((cal1 == 1) & (anc1 == a)).astype(np.int8)
    expected_dense_sel[:, ai::(2 * len(selected))] = mut_counts
    # dosage counts for selected ancestry a
    dosage_counts = (anc0 == a).astype(np.int8) + (anc1 == a).astype(np.int8)
    expected_dense_sel[:, (len(selected) + ai)::(2 * len(selected))] = dosage_counts

print(f"Expected dense (selected) shape: {expected_dense_sel.shape}")
print(expected_dense_sel)

print("Reconstructing dense matrix from file (selected ancestries)...")
dense_from_file_sel = handler_sel_read.to_dense(n_threads=2)
print(f"Actual dense (selected) shape: {dense_from_file_sel.shape}")
print(dense_from_file_sel)
np.testing.assert_array_equal(dense_from_file_sel, expected_dense_sel)
print("  - Selected ancestries mode verified.")

# Cleanup
for fn in [output_filename, output_filename_sel]:
    if os.path.exists(fn):
        os.remove(fn)
        print(f"Cleaned up '{fn}'.")


