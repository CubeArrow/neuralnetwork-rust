use crate::matrix::Matrix;

pub fn cost(expected: &Matrix, actual: &Matrix) -> f32{
    assert_eq!(expected.rows, actual.rows);
    let mut result = 0.0;
    for i in 0..expected.rows {
        let mut x = 0.0;
        for j in 0..expected.cols{
            let temp = expected.values[i][j] - actual.values[i][j];
            x += temp * temp;
        }
        result += x / expected.rows as f32;
    }
    return result;
}