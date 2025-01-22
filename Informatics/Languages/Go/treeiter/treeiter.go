//go:build !solution

package treeiter

type Node[T any] interface {
	Left() *T
	Right() *T
}

func DoInOrder[T any](tree *T, exec func(*T)) {
	if tree == nil {
		return
	}
	var tmp any = *tree
	node := tmp.(Node[T])
	DoInOrder(node.Left(), exec)
	exec(tree)
	DoInOrder(node.Right(), exec)
}
