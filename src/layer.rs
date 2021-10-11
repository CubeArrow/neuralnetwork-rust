use std::fmt::{Debug, Formatter};

use crate::matrix::Matrix;
use serde::Serialize;
use serde::Deserialize;

#[derive(Clone, serde::Serialize, Deserialize)]
pub struct Layer {
    pub weights: Matrix,
    pub biases: Matrix,
}

impl Layer {
    pub fn new(input_count: usize, output_count: usize) -> Result<Layer, String> {
        let weights = Matrix::new_random(input_count, output_count)?;
        let biases = Matrix::new_random(1, output_count)?;
        return Ok(Layer { weights, biases });
    }
    pub fn get_result(&self, input: &Matrix, activation_function: &dyn Fn(f32) -> f32) -> Result<Matrix, String> {
        let mut result = Matrix::matrix_multiplication(input, &self.weights)?;
        result = Matrix::matrix_addition_filling_rows(&result, &self.biases)?;
        result = result.apply_function(activation_function);
        return Ok(result);
    }
}

impl Debug for Layer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Weights:\n{:?}\n\nBiases:\n{:?}\n\n", self.weights, self.biases)
    }
}