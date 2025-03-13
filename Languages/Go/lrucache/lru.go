//go:build !solution

package lrucache

import (
	"container/list"
)

type elem struct {
	key   int
	value int
}

type cache struct {
	cap int
	l   *list.List
}

func (c cache) Search(key int) *list.Element {
	for e := c.l.Front(); e != nil; e = e.Next() {
		if e.Value.(*elem).key == key {
			return e
		}
	}
	return nil
}

func (c cache) Get(key int) (res int, ok bool) {
	i := c.Search(key)
	if i == nil {
		return 0, false
	}
	c.l.MoveToFront(i)
	return i.Value.(*elem).value, true
}

func (c cache) Set(key, value int) {
	i := c.Search(key)
	if i == nil {
		c.l.PushFront(&elem{key, value})
	} else {
		i.Value.(*elem).value = value
		c.l.MoveToFront(i)
	}
	if c.l.Len() > c.cap {
		c.l.Remove(c.l.Back())
	}
}

func (c cache) Range(f func(key, value int) bool) {
	for e := c.l.Back(); e != nil; e = e.Prev() {
		if !f(e.Value.(*elem).key, e.Value.(*elem).value) {
			break
		}
	}
}

func (c cache) Clear() {
	c.l.Init()
}

func New(cap int) Cache {
	return cache{cap, list.New()}
}
