//go:build !solution

package tparallel

import "sync"

type T struct {
	parent    *T
	waitGroup sync.WaitGroup
	suspended chan struct{}
	unlocked  chan struct{}
}

func (t *T) Parallel() {
	t.parent.suspended <- struct{}{}
	<-t.parent.unlocked
	t.parent.unlocked <- struct{}{}
}

func (t *T) Run(subtest func(t *T)) {
	t.waitGroup.Add(1)

	runFunc := func() {
		var son T
		son.parent = t
		son.suspended = make(chan struct{}, 1)
		son.unlocked = make(chan struct{}, 1)

		subtest(&son)
		son.unlocked <- struct{}{}
		son.waitGroup.Wait()

		t.waitGroup.Done()
		select {
		case <-t.suspended:
		default:
		}
		t.suspended <- struct{}{}
	}

	go runFunc()
	select {
	case <-t.suspended:
	}
}

func Run(topTests []func(t *T)) {
	var t T
	t.suspended = make(chan struct{}, 1)
	t.unlocked = make(chan struct{}, 1)

	for index := range topTests {
		t.Run(topTests[index])
	}
	t.unlocked <- struct{}{}
	t.waitGroup.Wait()
}
