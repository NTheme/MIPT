//go:build !solution

package hotelbusiness

import (
	"sort"
)

type Guest struct {
	CheckInDate  int
	CheckOutDate int
}

type Load struct {
	StartDate  int
	GuestCount int
}

type Event struct {
	Date int
	Type int
}

func GuestToEvents(guests []Guest) []Event {
	var events []Event
	for _, value := range guests {
		events = append(events, Event{value.CheckInDate, 1})
		events = append(events, Event{value.CheckOutDate, -1})
	}
	return events
}

func CountLoad(events []Event) []Load {
	var loaded []Load

	if len(events) > 0 {
		var counter = events[0].Type
		var lastLoad = 0

		events = append(events, Event{events[len(events)-1].Date + 1, 0})

		for index := 1; index < len(events); index++ {
			if events[index-1].Date != events[index].Date && lastLoad != counter {
				loaded = append(loaded, Load{events[index-1].Date, counter})
				lastLoad = loaded[len(loaded)-1].GuestCount
			}

			counter += events[index].Type
		}
	}
	return loaded
}

func ComputeLoad(guests []Guest) []Load {
	var events = GuestToEvents(guests)
	sort.Slice(events, func(i, j int) (less bool) {
		return events[i].Date < events[j].Date
	})
	return CountLoad(events)
}

