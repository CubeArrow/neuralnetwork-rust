use crate::matrix::Matrix;
use std::fmt::{Formatter, Debug};

#[derive(Clone)]
pub struct Layer {
    pub weights: Matrix,
    pub biases: Matrix,
}
impl Layer{
    pub fn new(input_count: usize, output_count: usize) -> Result<Layer, String>{
        let weights = Matrix::new_random(output_count, input_count)?;
        let biases = Matrix::new_random(output_count, 1)?;
        return Ok(Layer{weights, biases});
    }
    pub fn get_result(&self, input: &Matrix, activation_function: &dyn Fn(f32) -> f32) -> Result<Matrix, String>{
        let mut result = Matrix::matrix_multiplication(&self.weights, input)?;
        result = Matrix::matrix_addition(&result, &self.biases)?;
        result = result.apply_function(activation_function);
        return Ok(result)
    }
    pub fn change_bias(&mut self, added: Matrix){
        self.biases = Matrix::matrix_subtraction(&self.biases, &added).unwrap()
    }
}
impl Debug for Layer{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Weights:\n{:?}\n\nBiases:\n{:?}\n\n", self.weights, self.biases)
    }
}