use std::fmt::{Formatter, Debug};

pub struct Matrix {
    pub(crate) values: Vec<Vec<f32>>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl Matrix {
    pub fn new_zeroed(rows: usize, cols: usize) -> Result<Matrix, String> {
        if rows == 0 || cols == 0 {
            return Err("The dimensions may not be 0.".parse().unwrap());
        }
        let mut values = Vec::with_capacity(rows);
        for i in 0..rows {
            values.push(Vec::with_capacity(cols));
            for _ in 0..cols {
                values[i].push(0.0);
            }
        }
        return Ok(Matrix { values, rows, cols });
    }
    pub fn new_random(rows: usize, cols: usize) -> Result<Matrix, String> {
        if rows == 0 || cols == 0 {
            return Err("The dimensions may not be 0.".parse().unwrap());
        }
        let mut values = Vec::with_capacity(rows);
        for i in 0..rows {
            values.push(Vec::with_capacity(cols));
            for _ in 0..cols {
                values[i].push(rand::random::<f32>() * 2.0_f32 - 1.0);
            }
        }
        return Ok(Matrix { values, rows, cols });
    }

    pub fn scalar_multiplication(matrix: Matrix, scalar: f32) -> Matrix {
        let mut values = Vec::with_capacity(matrix.rows);
        for i in 0..matrix.rows {
            values.push(Vec::with_capacity(matrix.cols));
            for j in 0..matrix.cols {
                values[i].push(matrix.values[i][j] * scalar);
            }
        }
        return Matrix { values, rows: matrix.rows, cols: matrix.cols };
    }

    pub fn scalar_addition(matrix: Matrix, scalar: f32) -> Matrix {
        let mut values = Vec::with_capacity(matrix.rows);
        for i in 0..matrix.rows {
            values.push(Vec::with_capacity(matrix.cols));
            for j in 0..matrix.cols {
                values[i].push(matrix.values[i][j] + scalar);
            }
        }
        return Matrix { values, rows: matrix.rows, cols: matrix.cols };
    }
    pub fn matrix_multiplication(matrix: &Matrix, matrix2: &Matrix) -> Result<Matrix, String> {
        if matrix.cols != matrix2.rows {
            return Err("The column count of the left matrix is not equal to the row count of the right matrix.".parse().unwrap());
        }
        let rows = matrix.rows;
        let cols = matrix2.cols;

        let mut values = Vec::with_capacity(rows);

        for i in 0..rows {
            values.push(Vec::with_capacity(cols));
            for j in 0..cols {
                values[i].push(0.0);
                for k in 0..matrix.cols {
                    values[i][j] += matrix.values[i][k] * matrix2.values[k][j];
                }
            }
        }
        return Ok(Matrix { values, rows: matrix.rows, cols: matrix2.cols });
    }
    pub fn matrix_component_multiplication(matrix: &Matrix, matrix2: &Matrix) -> Result<Matrix, String> {
        let mut values = Vec::with_capacity(matrix.rows);

        for i in 0..matrix.rows {
            values.push(Vec::with_capacity(matrix.cols));
            for j in 0..matrix.cols {
                values[i].push(matrix.values[i][j] * matrix2.values[i][j]);
            }
        }
        return Ok(Matrix { values, rows: matrix.rows, cols: matrix.cols });
    }


    pub fn matrix_addition(matrix: &Matrix, matrix2: &Matrix) -> Result<Matrix, String> {
        if matrix.rows != matrix2.rows || matrix.cols != matrix2.cols {
            return Err("The dimensions of the two matrices do not match.".parse().unwrap());
        }
        let mut values = Vec::with_capacity(matrix.rows);
        for i in 0..matrix.rows {
            values.push(Vec::with_capacity(matrix.cols));
            for j in 0..matrix.cols {
                values[i].push(matrix.values[i][j] + matrix2.values[i][j]);
            }
        }
        return Ok(Matrix {
            values,
            rows: matrix.rows,
            cols: matrix.cols,
        });
    }


    pub fn matrix_subtraction(matrix: &Matrix, matrix2: &Matrix) -> Result<Matrix, String> {
        if matrix.rows != matrix2.rows || matrix.cols != matrix2.cols {
            return Err("The dimensions of the two matrices do not match.".parse().unwrap());
        }
        let mut values = Vec::with_capacity(matrix.rows);
        for i in 0..matrix.rows {
            values.push(Vec::with_capacity(matrix.cols));
            for j in 0..matrix.cols {
                values[i].push(matrix.values[i][j] - matrix2.values[i][j]);
            }
        }
        return Ok(Matrix {
            values,
            rows: matrix.rows,
            cols: matrix.cols,
        });
    }

    pub fn transpose(&self) -> Matrix {
        let mut values = Vec::with_capacity(self.cols);
        for i in 0..self.cols {
            values.push(Vec::with_capacity(self.rows));
            for j in 0..self.rows {
                values[i].push(self.values[j][i]);
            }
        }
        return Matrix { values, rows: self.cols, cols: self.rows };
    }
    pub fn apply_function(&self, function: &dyn Fn(f32) -> f32) -> Matrix{
        let mut values = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            values.push(Vec::with_capacity(self.cols));
            for j in 0..self.cols {
                values[i].push(function(self.values[i][j]));
            }
        }
        return Matrix { values, rows: self.rows, cols: self.cols };
    }

}
impl Debug for Matrix{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.values)
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Self {
        let mut values = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            values.push(Vec::with_capacity(self.cols));
            for j in 0..self.cols {
                values[i].push(self.values[i][j]);
            }
        }
        return Matrix { values, rows: self.rows, cols: self.cols };
    }
}
