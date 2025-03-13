//go:build !solution

package cond

// A Locker represents an object that can be locked and unlocked.
type Locker interface {
	Lock()
	Unlock()
}

// Cond implements a condition variable, a rendezvous point
// for goroutines waiting for or announcing the occurrence
// of an event.
//
// Each Cond has an associated Locker locker (often a *sync.Mutex or *sync.RWMutex),
// which must be held when changing the condition and
// when calling the Wait method.
type Cond struct {
	locker Locker
	waiter chan chan struct{}
}

// New returns a new Cond with Locker l.
func New(l Locker) *Cond {
	result := Cond{}
	result.locker = l
	result.waiter = make(chan chan struct{}, 1)
	result.waiter <- make(chan struct{}, 1)
	return &result
}

// Wait atomically unlocks c.locker and suspends execution
// of the calling goroutine. After later resuming execution,
// Wait locks c.locker before returning. Unlike in other systems,
// Wait cannot return unless awoken by Broadcast or Signal.
//
// Because c.locker is not locked when Wait first resumes, the caller
// typically cannot assume that the condition is true when
// Wait returns. Instead, the caller should Wait in a loop:
//
//	c.locker.Lock()
//	for !condition() {
//	    c.Wait()
//	}
//	... make use of condition ...
//	c.locker.Unlock()
func (c *Cond) Wait() {
	waiter := <-c.waiter
	c.locker.Unlock()
	c.waiter <- waiter
	<-waiter
	c.locker.Lock()
}

// Signal wakes one goroutine waiting on c, if there is any.
//
// It is allowed but not required for the caller to hold c.locker
// during the call.
func (c *Cond) Signal() {
	waiter := <-c.waiter
	select {
	case waiter <- struct{}{}:
	default:
	}
	c.waiter <- waiter
}

// Broadcast wakes all goroutines waiting on c.
//
// It is allowed but not required for the caller to hold c.locker
// during the call.
func (c *Cond) Broadcast() {
	close(<-c.waiter)
	c.waiter <- make(chan struct{}, 1)
}
