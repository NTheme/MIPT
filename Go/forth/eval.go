//go:build !solution

package main

import (
	"errors"
	"slices"
	"strconv"
	"strings"
)

type Evaluator struct {
	stack      []int
	definition map[string]string
}

// NewEvaluator creates evaluator.
func NewEvaluator() *Evaluator {
	return &Evaluator{
		stack:      make([]int, 0),
		definition: make(map[string]string),
	}
}

func (e *Evaluator) ProcessEval(row string, dfs bool) ([]int, error) {
	words := strings.Fields(row)

forLoop:
	for _, token := range words {
		token = strings.ToLower(token)
		if definition, ok := e.definition[strings.ToLower(token)]; ok && dfs {
			_, err := e.ProcessEval(definition, false)
			if err != nil {
				return nil, err
			}
			continue
		}
		switch token {
		case "+":
			if len(e.stack) < 2 {
				return nil, errors.New("+ exception")
			}
			op1 := e.stack[len(e.stack)-1]
			op2 := e.stack[len(e.stack)-2]
			e.stack = e.stack[:len(e.stack)-2]
			e.stack = append(e.stack, op1+op2)
		case "-":
			if len(e.stack) < 2 {
				return nil, errors.New("- exception")
			}
			op1 := e.stack[len(e.stack)-1]
			op2 := e.stack[len(e.stack)-2]
			e.stack = e.stack[:len(e.stack)-2]
			e.stack = append(e.stack, op2-op1)
		case "*":
			if len(e.stack) < 2 {
				return nil, errors.New("* exception")
			}
			op1 := e.stack[len(e.stack)-1]
			op2 := e.stack[len(e.stack)-2]
			e.stack = e.stack[:len(e.stack)-2]
			e.stack = append(e.stack, op1*op2)
		case "/":
			if len(e.stack) < 2 {
				return nil, errors.New("/ exception")
			}
			op1 := e.stack[len(e.stack)-1]
			op2 := e.stack[len(e.stack)-2]
			if op1 == 0 {
				return nil, errors.New("division by zero")
			}
			e.stack = e.stack[:len(e.stack)-2]
			e.stack = append(e.stack, op2/op1)
		case "dup":
			if len(e.stack) < 1 {
				return nil, errors.New("dup exception")
			}
			e.stack = append(e.stack, e.stack[len(e.stack)-1])
		case "over":
			if len(e.stack) < 2 {
				return nil, errors.New("over exception")
			}
			e.stack = append(e.stack, e.stack[len(e.stack)-2])
		case "drop":
			if len(e.stack) < 1 {
				return nil, errors.New("drop exception")
			}
			e.stack = e.stack[:len(e.stack)-1]
		case "swap":
			if len(e.stack) < 2 {
				return nil, errors.New("swap exception")
			}
			e.stack[len(e.stack)-1], e.stack[len(e.stack)-2] = e.stack[len(e.stack)-2], e.stack[len(e.stack)-1]
		case ":":
			if len(words) < 4 || words[len(words)-1] != ";" {
				return nil, errors.New("definition exception")
			}
			if _, err := strconv.Atoi(words[1]); err == nil {
				return nil, errors.New("definition number exception")
			}

			word := strings.ToLower(words[1])
			for j := 2; j < len(words)-1; j++ {
				if addDefinition, addOk := e.definition[strings.ToLower(words[j])]; addOk {
					words = slices.Concat(words[:j], []string{addDefinition}, words[j+1:])
					j--
				}
			}
			e.definition[word] = strings.Join(words[2:len(words)-1], " ")
			break forLoop
		default:
			num, err := strconv.Atoi(token)
			if err != nil {
				return nil, errors.New("unknown token: " + token)
			}
			e.stack = append(e.stack, num)
		}

	}
	return e.stack, nil
}

// Process evaluates sequence of words or definition.
//
// Returns resulting stack state and an error.
func (e *Evaluator) Process(row string) ([]int, error) {
	return e.ProcessEval(row, true)
}
