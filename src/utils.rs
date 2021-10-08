use crate::matrix::Matrix;

pub fn cost(expected: Matrix, actual: Matrix) -> f32{
    assert!(expected.cols == 1 && actual.cols == 1 && expected.rows == actual.rows);
    let mut result = 0.0;
    for i in 0..expected.rows {
        let x = expected.values[i][0] - actual.values[i][0];
        result += x * x;
    }
    return result;
}