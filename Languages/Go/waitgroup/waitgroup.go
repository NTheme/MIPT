//go:build !solution

package waitgroup

type RWMutex struct {
	lock  chan struct{}
	state chan int
}

func CreateRWMutex() *RWMutex {
	result := RWMutex{make(chan struct{}, 1), make(chan int, 1)}
	return &result
}

func (rw *RWMutex) RLock() {
	select {
	case rw.lock <- struct{}{}:
		rw.state <- 1
	case curr := <-rw.state:
		rw.state <- curr + 1
	}
}

func (rw *RWMutex) RUnlock() {
	curr := <-rw.state
	if curr > 1 {
		rw.state <- curr - 1
	} else {
		<-rw.lock
	}
}

func (rw *RWMutex) Lock() {
	rw.lock <- struct{}{}
}

func (rw *RWMutex) Unlock() {
	<-rw.lock
}

// A WaitGroup waits for a collection of goroutines to finish.
// The main goroutine calls Add to set the number of
// goroutines to wait for. Then each of the goroutines
// runs and calls Done when finished. At the same time,
// Wait can be used to block until all goroutines have finished.
type WaitGroup struct {
	mutex   *RWMutex
	counter int
	waiter  chan struct{}
}

// New creates WaitGroup.,
func New() *WaitGroup {
	result := WaitGroup{CreateRWMutex(), 0, make(chan struct{})}
	return &result
}

// Add adds delta, which may be negative, to the WaitGroup counter.
// If the counter becomes zero, all goroutines blocked on Wait are released.
// If the counter goes negative, Add panics.
//
// Note that calls with a positive delta that occur when the counter is zero
// must happen before a Wait. Calls with a negative delta, or calls with a
// positive delta that start when the counter is greater than zero, may happen
// at any time.
// Typically, this means the calls to Add should execute before the statement
// creating the goroutine or other event to be waited for.
// If a WaitGroup is reused to wait for several independent sets of events,
// new Add calls must happen after all previous Wait calls have returned.
// See the WaitGroup example.
func (wg *WaitGroup) Add(delta int) {
	wg.mutex.Lock()
	wg.counter += delta

	if wg.counter < 0 {
		panic("negative WaitGroup counter")
	}
	if wg.counter == 0 {
		close(wg.waiter)
	}
	if wg.counter == delta {
		wg.waiter = make(chan struct{})
	}

	wg.mutex.Unlock()
}

// Done decrements the WaitGroup counter by one.
func (wg *WaitGroup) Done() {
	wg.Add(-1)
}

// Wait blocks until the WaitGroup counter is zero.
func (wg *WaitGroup) Wait() {
	wg.mutex.RLock()
	waiter := wg.waiter
	wg.mutex.RUnlock()
	<-waiter
}
