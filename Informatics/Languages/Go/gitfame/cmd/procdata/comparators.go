package procdata

func LinesComparator(a Committer, b Committer) bool {
	if a.Lines != b.Lines {
		return a.Lines > b.Lines
	}
	if a.Commits != b.Commits {
		return a.Commits > b.Commits
	}
	if a.Touches != b.Touches {
		return a.Touches > b.Touches
	}
	return a.Name < b.Name
}

func CommitsComparator(a Committer, b Committer) bool {
	if a.Commits != b.Commits {
		return a.Commits > b.Commits
	}
	if a.Lines != b.Lines {
		return a.Lines > b.Lines
	}
	if a.Touches != b.Touches {
		return a.Touches > b.Touches
	}
	return a.Name < b.Name
}

func FilesComparator(a Committer, b Committer) bool {
	if a.Touches != b.Touches {
		return a.Touches > b.Touches
	}
	if a.Lines != b.Lines {
		return a.Lines > b.Lines
	}
	if a.Commits != b.Commits {
		return a.Commits > b.Commits
	}
	return a.Name < b.Name
}
