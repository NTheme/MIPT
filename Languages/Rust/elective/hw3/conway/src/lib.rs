#![forbid(unsafe_code)]

////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, PartialEq, Eq)]
pub struct Grid<T> {
    rows: usize,
    cols: usize,
    grid: Vec<T>,
}

pub struct Neighbours {
    rows: usize,
    cols: usize,
    idx: usize,
    x: usize,
    y: usize,
}

impl Iterator for Neighbours {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        while self.idx < 9 {
            let dx = self.idx / 3;
            let dy = self.idx % 3;
            self.idx += 1;

            if dx == 1 && dy == 1 {
                continue;
            }

            let x = self.x + dx;
            let y = self.y + dy;
            if x >= 1 && x <= self.rows && y >= 1 && y <= self.cols {
                return Some((x - 1, y - 1));
            }
        }
        None
    }
}

impl<T: Clone + Default> Grid<T> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            grid: vec![T::default(); rows * cols],
        }
    }

    pub fn from_slice(grid: &[T], rows: usize, cols: usize) -> Self {
        assert_eq!(grid.len(), rows * cols);
        Self {
            rows,
            cols,
            grid: Vec::from(grid),
        }
    }

    pub fn size(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn get(&self, row: usize, col: usize) -> &T {
        &self.grid[row * self.cols + col]
    }

    pub fn set(&mut self, value: T, row: usize, col: usize) {
        self.grid[row * self.cols + col] = value;
    }

    pub fn neighbours(&self, row: usize, col: usize) -> Neighbours {
        Neighbours {
            rows: self.rows,
            cols: self.cols,
            idx: 0,
            x: row,
            y: col,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    Dead,
    Alive,
}

impl Default for Cell {
    fn default() -> Self {
        Self::Dead
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(PartialEq, Eq)]
pub struct GameOfLife {
    grid: Grid<Cell>,
}

impl GameOfLife {
    pub fn from_grid(grid: Grid<Cell>) -> Self {
        Self { grid }
    }

    pub fn get_grid(&self) -> &Grid<Cell> {
        &self.grid
    }

    pub fn step(&mut self) {
        let mut next = vec![];

        for i in 0..self.grid.size().0 {
            for j in 0..self.grid.size().1 {
                let mut alive = 0;
                for (x, y) in self.grid.neighbours(i, j) {
                    if *self.grid.get(x, y) == Cell::Alive {
                        alive += 1
                    }
                }

                next.push(match alive {
                    3 => Cell::Alive,
                    2 if *self.grid.get(i, j) == Cell::Alive => Cell::Alive,
                    _ => Cell::Dead,
                })
            }
        }
        self.grid.grid = next;
    }
}
