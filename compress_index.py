import faiss
import time
import os
import numpy as np

def compress_to_8bit():
    original_path = "faiss_index.index"
    compressed_path = "faiss_index_8bit.index"
    
    if not os.path.exists(original_path):
        print(f"Error: Could not find {original_path}. Please ensure it exists in the current directory.")
        return

    print(f"1. Loading original Float32 index from {original_path} (614 MB)...")
    start = time.time()
    index = faiss.read_index(original_path)
    print(f"   Loaded {index.ntotal} vectors in {time.time() - start:.2f}s")
    
    dim = index.d
    total = index.ntotal
    
    print("2. Extracting vectors into memory...")
    # Extract all vectors from the FlatL2 index
    # We must do this in chunks if memory is limited, but 614MB is fine for local machine
    vectors = np.zeros((total, dim), dtype=np.float32)
    for i in range(total):
        vectors[i] = index.reconstruct(i)
    
    print("3. Creating 8-bit Scalar Quantizer Index...")
    # QT_8bit compresses each float32 (4 bytes) into 1 byte -> 75% memory reduction
    sq_index = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2)
    
    print("4. Training the quantizer (finding min/max for compression)...")
    # Train on a sample of vectors to find bounds
    sq_index.train(vectors)
    
    print("5. Adding vectors to the new 8-bit index...")
    sq_index.add(vectors)
    
    print(f"6. Saving compressed index to {compressed_path}...")
    faiss.write_index(sq_index, compressed_path)
    
    orig_size = os.path.getsize(original_path) / (1024*1024)
    comp_size = os.path.getsize(compressed_path) / (1024*1024)
    
    print("=========================================")
    print(f"✅ Compression Complete!")
    print(f"Original Size:   {orig_size:.1f} MB")
    print(f"Compressed Size: {comp_size:.1f} MB ({(1 - comp_size/orig_size)*100:.1f}% reduction)")
    print("=========================================")

if __name__ == "__main__":
    compress_to_8bit()
