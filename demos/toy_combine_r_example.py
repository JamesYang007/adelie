import numpy as np
import adelie.io as io
import adelie.data as data
import os
import time

# This example demonstrates how to create a .snpdat file for 
# SNP unphased ancestry data, write it to a file, read it back,
# and verify the contents.

# Set a seed for reproducibility
np.random.seed(0)

# Toy data parameters
n_samples = 5
n_snps = 7
n_ancestries = 3

print("Generating toy data with parameters:")
print(f"  - Samples: {n_samples}")
print(f"  - SNPs: {n_snps}")
print(f"  - Ancestries: {n_ancestries}")

# Generate toy data using snp_combine_r
data_dict = data.snp_combine_r(
    n=n_samples,
    s=n_snps,
    A=n_ancestries,
    seed=0
)

# Extract calldata and ancestries from the generated data
calldata = data_dict["X"]
ancestries = data_dict["ancestries"]

print(f"\nCalldata matrix (shape: {calldata.shape}):")
print(calldata)

print(f"\nAncestries matrix (shape: {ancestries.shape}):")
print(ancestries)

# Benchmark the nested for loops approach
print("\nBenchmarking nested for loops approach...")
start_time = time.time()

# For verification, create the expected dense matrix from original data.
# The structure of the dense matrix is explained in the docstring of
# adelie.io.snp_combine_r.
expected_dense = np.zeros((n_samples, n_snps * (1 + n_ancestries)), dtype=np.int8)
for i in range(n_samples):
    for j in range(n_snps):
        # First column of each SNP block is the calldata value
        expected_dense[i, j * (1 + n_ancestries)] = calldata[i, j]
        
        # Next A columns are ancestry dosages
        # For each ancestry type, count how many haplotypes have that ancestry
        for a in range(n_ancestries):
            # Count ancestries for both haplotypes at this SNP position
            if ancestries[i, 2*j] == a:
                expected_dense[i, j * (1 + n_ancestries) + 1 + a] += 1
            if ancestries[i, 2*j + 1] == a:
                expected_dense[i, j * (1 + n_ancestries) + 1 + a] += 1

nested_loops_time = time.time() - start_time
print(f"Nested loops approach took: {nested_loops_time:.6f} seconds")

# Vectorized approach
print("\nBenchmarking vectorized approach...")
start_time = time.time()

# Create the dense matrix using vectorized operations
expected_dense_vectorized = np.zeros((n_samples, n_snps * (1 + n_ancestries)), dtype=np.int8)

# Set the calldata values (first column of each SNP block)
expected_dense_vectorized[:, ::(1 + n_ancestries)] = calldata

# Create ancestry counts using broadcasting
for a in range(n_ancestries):
    # Count ancestries for both haplotypes at each SNP position
    ancestry_counts = (ancestries[:, ::2] == a).astype(np.int8) + (ancestries[:, 1::2] == a).astype(np.int8)
    # Place counts in the appropriate columns
    expected_dense_vectorized[:, a+1::(1 + n_ancestries)] = ancestry_counts

vectorized_time = time.time() - start_time
print(f"Vectorized approach took: {vectorized_time:.6f} seconds")

# Verify both approaches produce the same result
np.testing.assert_array_equal(expected_dense, expected_dense_vectorized)
print("Verification successful: vectorized approach matches nested loops approach")

print(f"\nExpected dense matrix (shape: {expected_dense.shape}):")
print(expected_dense)

# Define output file path
output_filename = "combine_r.snpdat"

# Create an IO handler for SNP unphased ancestry data.
handler = io.snp_combine_r(filename=output_filename)

# Write the data to the file in .snpdat format.
print(f"\nWriting data to '{output_filename}'...")
try:
    # Benchmark the io.snp_combine_r approach
    print("\nBenchmarking io.snp_combine_r approach...")
    start_time = time.time()
    total_bytes, benchmark = handler.write(
        calldata=calldata,
        ancestries=ancestries,
        A=n_ancestries,
        n_threads=2  # Example of using 2 threads
    )

    print(f"Successfully wrote {total_bytes} bytes.")
    print("Benchmark timings for writing:")
    for key, value in benchmark.items():
        print(f"  - {key}: {value:.6f} seconds")

    # Now, demonstrate reading the data back.
    print(f"\nReading data back from '{output_filename}'...")
    
    # Create a new handler for reading.
    handler_read = io.snp_combine_r(filename=output_filename, read_mode="mmap")
    
    handler_read.read()

    # Verify the properties of the read data.
    print("Verifying properties of the read data...")
    assert handler_read.is_read
    assert handler_read.rows == n_samples
    assert handler_read.snps == n_snps
    assert handler_read.ancestries == n_ancestries
    assert handler_read.cols == n_snps * (1 + n_ancestries)  # Each SNP has 1 + n_ancestries columns
    print("  - Properties (rows, snps, ancestries, cols) are correct.")

    # Reconstruct the dense matrix from the file.
    # The to_dense() method returns a matrix of shape (n_samples, n_snps * (1 + n_ancestries)).
    dense_matrix_from_file = handler_read.to_dense(n_threads=2)
    io_approach_time = time.time() - start_time
    print(f"IO approach took: {io_approach_time:.6f} seconds")
    
    print(f"\nDense matrix from file (shape: {dense_matrix_from_file.shape}):")
    print(dense_matrix_from_file)
    
    # Compare the reconstructed matrix with the expected one.
    np.testing.assert_array_equal(dense_matrix_from_file, expected_dense)
    print("  - Verification successful: reconstructed matrix matches expected matrix.")

    # ------------------------------------------------------------------
    # Demonstrate selected ancestries (e.g., only ancestries 0 and 2)
    # ------------------------------------------------------------------
    selected = np.array([0, 2], dtype=np.uint32)
    output_filename_sel = "combine_r_selected.snpdat"
    handler_sel = io.snp_combine_r(filename=output_filename_sel)
    # Write only selected ancestries; output has (1 + len(selected)) columns per SNP
    total_bytes_sel, benchmark_sel = handler_sel.write(
        calldata=calldata,
        ancestries=ancestries,
        A=n_ancestries,
        n_threads=2,
        selected_ancestries=selected
    )

    handler_sel_read = io.snp_combine_r(filename=output_filename_sel, read_mode="mmap")
    handler_sel_read.read()

    assert handler_sel_read.rows == n_samples
    assert handler_sel_read.snps == n_snps
    assert handler_sel_read.ancestries == len(selected)
    assert handler_sel_read.cols == n_snps * (1 + len(selected))

    # Build expected dense restricted to selected ancestries
    expected_dense_sel = np.zeros((n_samples, n_snps * (1 + len(selected))), dtype=np.int8)
    # Genotype column stays identical
    expected_dense_sel[:, ::(1 + len(selected))] = calldata
    # Ancestry dosage columns only for selected labels, in the same order as 'selected'
    for ai, a in enumerate(selected):
        ancestry_counts = (ancestries[:, ::2] == a).astype(np.int8) + (ancestries[:, 1::2] == a).astype(np.int8)
        expected_dense_sel[:, ai+1::(1 + len(selected))] = ancestry_counts

    dense_matrix_from_file_sel = handler_sel_read.to_dense(n_threads=2)
    
    # Print both matrices for comparison
    print(f"\nExpected dense matrix for selected ancestries {selected} (shape: {expected_dense_sel.shape}):")
    print(expected_dense_sel)
    
    print(f"\nActual dense matrix from file for selected ancestries {selected} (shape: {dense_matrix_from_file_sel.shape}):")
    print(dense_matrix_from_file_sel)
    
    np.testing.assert_array_equal(dense_matrix_from_file_sel, expected_dense_sel)
    print("  - Selected ancestries mode verified.")
    
    # Print performance comparison
    print("\nPerformance Comparison:")
    print(f"Nested loops approach: {nested_loops_time:.6f} seconds")
    print(f"Vectorized approach: {vectorized_time:.6f} seconds")
    print(f"IO approach: {io_approach_time:.6f} seconds")
    print(f"Speedup factor (vectorized vs nested): {nested_loops_time/vectorized_time:.2f}x")
    print(f"Speedup factor (IO vs nested): {nested_loops_time/io_approach_time:.2f}x")

finally:
    # Clean up the created file.
    if os.path.exists(output_filename):
        os.remove(output_filename)
        print(f"\nCleaned up '{output_filename}'.")
    if os.path.exists(output_filename_sel):
        os.remove(output_filename_sel)
        print(f"Cleaned up '{output_filename_sel}'.")
