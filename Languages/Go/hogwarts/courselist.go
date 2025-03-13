//go:build !solution

package hogwarts

type Vertex struct {
	CourseName string
	Children   []int
}

func DFS(course string, prereqs *map[string][]string, colors *map[string]int, way *[]string) {
	(*colors)[course] = 1
	*way = append(*way, course)

	for _, to := range (*prereqs)[course] {
		if (*colors)[to] == 0 {
			DFS(to, prereqs, colors, way)
		} else if (*colors)[to] == 1 {
			panic(1)
		}
	}

	(*colors)[course] = 2
}

func CoursesToTree(prereqs *map[string][]string) {

}

func GetCourseList(prereqs map[string][]string) []string {
	colors := make(map[string]int)
	for key, value := range prereqs {
		colors[key] = 0
		for _, course := range value {
			colors[course] = 0
		}
	}

	var courseList []string

	for course, color := range colors {
		if color == 0 {
			var add []string
			DFS(course, &prereqs, &colors, &add)

			for i, j := 0, len(add)-1; i < j; i, j = i+1, j-1 {
				add[i], add[j] = add[j], add[i]
			}
			courseList = append(courseList, add...)
		}
	}

	return courseList
}
