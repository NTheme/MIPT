//go:build !solution

package genericsum

import (
	"golang.org/x/exp/constraints"
	"math/cmplx"
	"sort"
)

func Min[T constraints.Ordered](a, b T) T {
	if a < b {
		return a
	}
	return b
}

func SortSlice[T constraints.Ordered](a []T) {
	sort.Slice(a, func(i, j int) bool {
		return a[i] < a[j]
	})
}

func MapsEqual[T1 comparable, T2 comparable](a, b map[T1]T2) bool {
	if len(a) != len(b) {
		return false
	}
	for key, value := range a {
		val, ok := b[key]
		if !ok || val != value {
			return false
		}
	}
	return true
}

func SliceContains[T comparable](s []T, v T) bool {
	for i := range s {
		if s[i] == v {
			return true
		}
	}
	return false
}

func Merge[T any](res chan T, chs ...<-chan T) {
	for {
		closed := 0
		for i := range chs {
			select {
			case val, ok := <-chs[i]:
				if ok {
					res <- val
					continue
				}
				closed++
			default:
			}
		}
		if closed == len(chs) {
			break
		}
	}
	close(res)
}

func MergeChans[T any](chs ...<-chan T) <-chan T {
	res := make(chan T, len(chs))
	go Merge(res, chs...)
	return res
}

type number interface {
	constraints.Complex | constraints.Float | constraints.Integer
}

func IsHermitianMatrix[T number](a [][]T) bool {
	for i := 0; i < len(a); i++ {
		for j := 0; j < len(a[0]); j++ {
			var value1 any = a[i][j]
			var value2 any = a[j][i]

			switch value1.(type) {
			case complex128:
				if value1.(complex128) != cmplx.Conj(value2.(complex128)) {
					return false
				}
			case complex64:
				if (complex128)(value1.(complex64)) != cmplx.Conj((complex128)(value2.(complex64))) {
					return false
				}
			default:
				if value1 != value2 {
					return false
				}
			}
		}
	}
	return true
}
