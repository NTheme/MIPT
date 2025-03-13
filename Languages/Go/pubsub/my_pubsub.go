//go:build !solution

package pubsub

import (
	"context"
	"fmt"
	"sync"
)

var _ Subscription = (*MySubscription)(nil)

type MySubscription struct {
	message chan any
	locker  chan struct{}
}

func (s *MySubscription) Unsubscribe() {
	close(s.locker)
	close(s.message)
}

var _ PubSub = (*MyPubSub)(nil)

type MyPubSub struct {
	mutex         sync.Mutex
	locker        chan struct{}
	subscriptions map[string][]*MySubscription
}

func NewPubSub() PubSub {
	res := MyPubSub{sync.Mutex{}, make(chan struct{}), make(map[string][]*MySubscription)}
	return &res
}

func DeferLocker(p *MyPubSub) error {
	select {
	case <-p.locker:
		return fmt.Errorf("")
	default:
	}
	return nil
}

func Message(vv *MySubscription, msg *interface{}) {
	defer func() { recover() }()
	vv.message <- *msg
}

func Recover() {
	recover()
}

func Close(message *chan any) {
	defer Recover()
	close(*message)
}

func EvaluateSub(s *MySubscription, cb *MsgHandler) {
	for {
		v, err := <-s.message
		if !err {
			return
		}
		(*cb)(v)
	}
}

func (p *MyPubSub) Subscribe(subj string, cb MsgHandler) (Subscription, error) {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	if DeferLocker(p) != nil {
		return nil, fmt.Errorf("")
	}

	if p.subscriptions[subj] == nil {
		p.subscriptions[subj] = make([]*MySubscription, 0, 1)
	}

	subscription := MySubscription{make(chan any, 500), make(chan struct{})}
	p.subscriptions[subj] = append(p.subscriptions[subj], &subscription)

	go EvaluateSub(&subscription, &cb)
	return &subscription, nil
}

func (p *MyPubSub) Publish(subj string, msg interface{}) error {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	if DeferLocker(p) != nil {
		return fmt.Errorf("")
	}

	for index := range len(p.subscriptions[subj]) {
		Message(p.subscriptions[subj][index], &msg)
	}

	return nil
}

func (p *MyPubSub) Close(ctx context.Context) error {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	if DeferLocker(p) != nil {
		return fmt.Errorf("")
	}

	close(p.locker)
	for _, subscription := range p.subscriptions {
		for j := range len(subscription) {
			Close(&subscription[j].message)
		}
	}
	return nil
}
