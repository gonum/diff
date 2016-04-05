// Copyright ©2016 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fd

import (
	"math"
	"math/rand"
	"testing"

	"github.com/gonum/matrix/mat64"
)

func quadratic(x []float64) float64                { return x[0] * x[0] }
func quadraticHess(h *mat64.SymDense, _ []float64) { h.SetSym(0, 0, 2) }

func saddle(x []float64) float64 {
	return x[0]*x[0] - x[1]*x[1]
}
func saddleHess(h *mat64.SymDense, x []float64) {
	h.SetSym(0, 0, 2)
	h.SetSym(1, 1, -2)
	h.SetSym(0, 1, 0)
}

func monkeySaddle(x []float64) float64 {
	return x[0]*x[0]*x[0] - 3*x[0]*x[1]*x[1]
}
func monkeySaddleHess(h *mat64.SymDense, x []float64) {
	h.SetSym(0, 0, 6*x[0])
	h.SetSym(1, 1, -6*x[0])
	h.SetSym(0, 1, -6*x[1])
}

func TestHessian(t *testing.T) {
	rand.Seed(1)
	for i, test := range []struct {
		name string
		dim  int
		f    func([]float64) float64
		hess func(h *mat64.SymDense, x []float64)
		tol  float64
	}{
		{
			name: "quadratic",
			dim:  1,
			f:    quadratic,
			hess: quadraticHess,
			tol:  1e-6,
		},
		{
			name: "standard saddle",
			dim:  2,
			f:    saddle,
			hess: saddleHess,
			tol:  1e-6,
		},
		{
			name: "monkey saddle",
			dim:  2,
			f:    monkeySaddle,
			hess: monkeySaddleHess,
			tol:  1e-4,
		},
	} {
		for k := 0; k < 10; k++ {
			x := randomSlice(test.dim, 10)
			want := mat64.NewSymDense(test.dim, nil)
			test.hess(want, x)
			got := Hessian(nil, test.f, x, nil)
			if !mat64.EqualApprox(want, got, test.tol) {
				t.Errorf("Case %d (%s, nil dst, default settings): unexpected Hessian:\nwant: %v\ngot:  %v",
					i, test.name, mat64.Formatted(want, mat64.Prefix("      ")), mat64.Formatted(got, mat64.Prefix("      ")))
			}

			fillNaNSym(got)
			Hessian(got, test.f, x, nil)
			if !mat64.EqualApprox(want, got, test.tol) {
				t.Errorf("Case %d (%s, default settings): unexpected Hessian:\nwant: %v\ngot:  %v",
					i, test.name, mat64.Formatted(want, mat64.Prefix("      ")), mat64.Formatted(got, mat64.Prefix("      ")))
			}

			fillNaNSym(got)
			Hessian(got, test.f, x, &HessianSettings{
				OriginKnown: true,
				OriginValue: test.f(x),
			})
			if !mat64.EqualApprox(want, got, test.tol) {
				t.Errorf("Case %d (%s, known origin): unexpected Hessian:\nwant: %v\ngot:  %v",
					i, test.name, mat64.Formatted(want, mat64.Prefix("      ")), mat64.Formatted(got, mat64.Prefix("      ")))
			}

			fillNaNSym(got)
			Hessian(got, test.f, x, &HessianSettings{
				Concurrent: true,
			})
			if !mat64.EqualApprox(want, got, test.tol) {
				t.Errorf("Case %d (%s, concurrent): unexpected Hessian:\nwant: %v\ngot:  %v",
					i, test.name, mat64.Formatted(want, mat64.Prefix("      ")), mat64.Formatted(got, mat64.Prefix("      ")))
			}

			fillNaNSym(got)
			Hessian(got, test.f, x, &HessianSettings{
				Concurrent:  true,
				OriginKnown: true,
				OriginValue: test.f(x),
			})
			if !mat64.EqualApprox(want, got, test.tol) {
				t.Errorf("Case %d (%s, known origin, concurrent): unexpected Hessian:\nwant: %v\ngot:  %v",
					i, test.name, mat64.Formatted(want, mat64.Prefix("      ")), mat64.Formatted(got, mat64.Prefix("      ")))
			}
		}
	}
}

// fillNaNSym fills the symmetric matrix m with NaN values.
func fillNaNSym(m *mat64.SymDense) {
	r, c := m.Dims()
	for i := 0; i < r; i++ {
		for j := i; j < c; j++ {
			m.SetSym(i, j, math.NaN())
		}
	}
}
