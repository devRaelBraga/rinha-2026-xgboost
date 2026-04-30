package main

import (
	"fmt"
	"os"
)

// KNNIndex holds the reference vectors and their labels in a compact format.
type KNNIndex struct {
	vectors []uint8  // 3,000,000 * 14 bytes = 42,000,000
	labels  []byte   // 3,000,000 bits = 375,000 bytes
	n       int
}

// LoadKNNIndex loads the preprocessed binary files into memory.
func LoadKNNIndex(vectorsPath, labelsPath string) (*KNNIndex, error) {
	vecData, err := os.ReadFile(vectorsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read vectors file: %w", err)
	}

	labelData, err := os.ReadFile(labelsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read labels file: %w", err)
	}

	n := len(vecData) / 14
	if n != 3000000 {
		fmt.Printf("Warning: Expected 3,000,000 vectors, found %d\n", n)
	}

	return &KNNIndex{
		vectors: vecData,
		labels:  labelData,
		n:       n,
	}, nil
}

// Search performs a brute-force KNN search (k=5) and returns the fraud score.
func (idx *KNNIndex) Search(query [14]float32, k int) float64 {
	// 1. Quantize query to uint8
	var q [14]uint8
	for i, v := range query {
		if v == -1.0 {
			q[i] = 255
		} else {
			// Round: v*250 + 0.5
			val := v*250.0 + 0.5
			if val < 0 {
				q[i] = 0
			} else if val > 250 {
				q[i] = 250
			} else {
				q[i] = uint8(val)
			}
		}
	}

	// 2. Linear scan for top-k nearest neighbors
	// We use a simple insertion sort for the top 5 distances.
	var topDist [5]uint32
	var topIdx [5]int
	for i := 0; i < 5; i++ {
		topDist[i] = 0xFFFFFFFF
	}

	vectors := idx.vectors
	n := idx.n

	for i := 0; i < n; i++ {
		offset := i * 14
		
		// Manual unrolling for 14 dimensions to help the compiler/CPU
		// Using int32 for subtraction to prevent underflow before squaring
		v0 := int32(vectors[offset]) - int32(q[0])
		v1 := int32(vectors[offset+1]) - int32(q[1])
		v2 := int32(vectors[offset+2]) - int32(q[2])
		v3 := int32(vectors[offset+3]) - int32(q[3])
		v4 := int32(vectors[offset+4]) - int32(q[4])
		v5 := int32(vectors[offset+5]) - int32(q[5])
		v6 := int32(vectors[offset+6]) - int32(q[6])
		v7 := int32(vectors[offset+7]) - int32(q[7])
		v8 := int32(vectors[offset+8]) - int32(q[8])
		v9 := int32(vectors[offset+9]) - int32(q[9])
		v10 := int32(vectors[offset+10]) - int32(q[10])
		v11 := int32(vectors[offset+11]) - int32(q[11])
		v12 := int32(vectors[offset+12]) - int32(q[12])
		v13 := int32(vectors[offset+13]) - int32(q[13])

		dist := uint32(v0*v0 + v1*v1 + v2*v2 + v3*v3 + v4*v4 + v5*v5 + v6*v6 + v7*v7 + v8*v8 + v9*v9 + v10*v10 + v11*v11 + v12*v12 + v13*v13)

		// Optimization: only update if better than the worst in our top-k
		if dist < topDist[4] {
			// Find position to insert
			pos := 4
			for pos > 0 && dist < topDist[pos-1] {
				topDist[pos] = topDist[pos-1]
				topIdx[pos] = topIdx[pos-1]
				pos--
			}
			topDist[pos] = dist
			topIdx[pos] = i
		}
	}

	// 3. Calculate fraud score (unweighted as per challenge rules)
	fraudCount := 0
	for i := 0; i < 5; i++ {
		idxVal := topIdx[i]
		byteIdx := idxVal / 8
		bitIdx := uint(idxVal % 8)
		if (idx.labels[byteIdx] & (1 << bitIdx)) != 0 {
			fraudCount++
		}
	}

	return float64(fraudCount) / 5.0
}
