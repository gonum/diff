// Copyright ©2016 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fd

import (
	"runtime"
	"sync"

	"github.com/gonum/matrix/mat64"
)

type HessianSettings struct {
	OriginKnown bool
	OriginValue float64
	Step        float64
	Concurrent  bool
}

func Hessian(dst *mat64.SymDense, f func([]float64) float64, x []float64, settings *HessianSettings) *mat64.SymDense {
	n := len(x)
	if dst == nil {
		dst = mat64.NewSymDense(n, nil)
	}
	if dst.Symmetric() != n {
		panic("hessian: mismatched matrix size")
	}

	if settings == nil {
		settings = &HessianSettings{}
	}

	step := settings.Step
	if step == 0 {
		step = Central2nd.Step
	}

	expect := n + n*(n-1)/2 // Diagonal + half of off-diagonal elements.
	nWorkers := 1
	if settings.Concurrent {
		nWorkers = runtime.GOMAXPROCS(0)
		if nWorkers > expect {
			nWorkers = expect
		}
	}

	xcopy := make([]float64, n)
	origin := settings.OriginValue
	if !settings.OriginKnown {
		copy(xcopy, x)
		origin = f(xcopy)
	}

	if nWorkers == 1 {
		hessianSerial(dst, f, x, xcopy, step, origin)
	} else {
		hessianConcurrent(dst, f, x, step, origin, nWorkers)
	}
	return dst
}

func hessianSerial(dst *mat64.SymDense, f func([]float64) float64, x, xcopy []float64, step, origin float64) {
	// Evaluate f at neighboring points so that neigh[i] = f(x + step * e_i).
	neigh := make([]float64, len(x))
	for i := range xcopy {
		copy(xcopy, x)
		xcopy[i] += step
		neigh[i] = f(xcopy)
	}
	for i := range xcopy {
		copy(xcopy, x)
		xcopy[i] -= step
		fii := f(xcopy)
		dst.SetSym(i, i, ((neigh[i]-origin)/step-(origin-fii)/step)/step)
		for j := i + 1; j < len(x); j++ {
			copy(xcopy, x)
			xcopy[i] += step
			xcopy[j] += step
			fij := f(xcopy)
			dst.SetSym(i, j, ((fij-neigh[j])/step-(neigh[i]-origin)/step)/step)
		}
	}
}

func hessianConcurrent(dst *mat64.SymDense, f func([]float64) float64, x []float64, step, origin float64, nWorkers int) {
	n := len(x)
	var wg sync.WaitGroup

	neigh := make([]float64, n)
	neighWorkers := nWorkers
	if neighWorkers > n {
		neighWorkers = n
	}
	neighJobs := make(chan int, neighWorkers)
	for i := 0; i < neighWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			xcopy := make([]float64, n)
			for job := range neighJobs {
				copy(xcopy, x)
				xcopy[job] += step
				neigh[job] = f(xcopy)
			}
		}()
	}
	for i := range neigh {
		neighJobs <- i
	}
	close(neighJobs)
	wg.Wait()

	jobs := make(chan hessJob, nWorkers)
	for i := 0; i < nWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			xcopy := make([]float64, n)
			for job := range jobs {
				copy(xcopy, x)
				if job.i == job.j {
					xcopy[job.i] -= step
				} else {
					xcopy[job.i] += step
					xcopy[job.j] += step
				}
				fx := f(xcopy)
				if job.i == job.j {
					dst.SetSym(job.i, job.i, (fx-origin+neigh[job.i]-origin)/step/step)
				} else {
					dst.SetSym(job.i, job.j, (fx-neigh[job.j]+origin-neigh[job.i])/step/step)
				}
			}
		}()
	}
	hessianProducer(jobs, n)
	wg.Wait()
}

func hessianProducer(jobs chan<- hessJob, n int) {
	for i := 0; i < n; i++ {
		jobs <- hessJob{
			i: i,
			j: i,
		}
		for j := i + 1; j < n; j++ {
			jobs <- hessJob{
				i: i,
				j: j,
			}
		}
	}
	close(jobs)
}

type hessJob struct {
	i, j int
}
