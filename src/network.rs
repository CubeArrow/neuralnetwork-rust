use std::fmt::{Debug, Formatter};

use crate::layer::Layer;
use crate::matrix::Matrix;

use serde::Serialize;
use serde::Deserialize;

#[derive(Clone, Serialize, Deserialize)]
pub struct Network {
    pub(crate) layers: Vec<Layer>,
}

impl Network {
    pub fn new(layer_sizes: &[usize]) -> Result<Network, String> {
        let mut layers = Vec::new();
        for i in 1..layer_sizes.len() {
            layers.push(Layer::new(layer_sizes[i - 1], layer_sizes[i])?)
        }
        return Ok(Network { layers });
    }
    pub fn feedforward(&self, input: Matrix, activaction_function: &dyn Fn(f32) -> f32) -> Result<Vec<Matrix>, String> {
        let mut res = vec![input];
        for i in 0..self.layers.len() {
            res.push(self.get_result_index(i, res.last().unwrap(), activaction_function)?);
        }
        return Ok(res);
    }
    pub fn get_result_index(&self, index: usize, input: &Matrix, activation_function: &dyn Fn(f32) -> f32) -> Result<Matrix, String> {
        return self.layers[index].clone().get_result(input, activation_function);
    }


    pub fn backpropagate(&mut self, result: &Vec<Matrix>, expected: Matrix, derivative_activation_function: &dyn Fn(f32) -> f32, learning_rate: f32) {
        let mut delta = Matrix::matrix_subtraction(&result.last().unwrap(), &expected).unwrap().transpose();

        for layer_index in (0..self.layers.len()).rev() {
            let mut delta_weights =  Matrix::matrix_multiplication(&delta,&result[layer_index]).unwrap().transpose();
            delta_weights.scalar_multiplication_mut(learning_rate);

            let mut delta_biases =
                Matrix::new_zeroed(1, self.layers[layer_index].biases.cols).unwrap();
            for i in 0..expected.rows {
                delta_biases.matrix_addition_mut(&delta.transpose().get_single_row(i));
            }
            delta_biases.scalar_multiplication_mut(learning_rate / expected.rows as f32);


            delta = Matrix::matrix_multiplication(
                        &self.layers[layer_index].weights,
                        &delta).unwrap();
            delta.matrix_component_multiplication_mut(&result[layer_index].apply_function(derivative_activation_function).transpose());

            self.layers.get_mut(layer_index).unwrap().weights.matrix_subtraction_mut(&delta_weights);
            self.layers.get_mut(layer_index).unwrap().biases.matrix_subtraction_mut(&delta_biases);
        }
    }
    // pub fn backpropagate2(&mut self, result: &Vec<Matrix>, expected: Matrix, derivative_activation_function: &dyn Fn(f32) -> f32, learning_rate: f32) {
    //     // Calculate the initial error
    //     let mut errors = Matrix::matrix_subtraction(&expected, &result.last().unwrap()).unwrap();
    //     for layer_index in (0..self.layers.len()).rev() {
    //         let mut delta_weights =
    //             Matrix::new_zeroed(self.layers[layer_index].weights.rows, self.layers[layer_index].weights.cols).unwrap();
    //         let mut delta_biases =
    //             Matrix::new_zeroed(self.layers[layer_index].biases.rows, 1).unwrap();
    //
    //         for i in 0..expected.cols {
    //             // Apply the derivative of the activation function to the output of the current layer
    //             let d_out_dnet = result[layer_index + 1].get_single_col(i).apply_function(derivative_activation_function);
    //
    //             // get the inputs of the current layer
    //             let d_net_w = &result[layer_index].get_single_col(i);
    //
    //             // Calculate the weights to be subtracted
    //             let weight_changes_without_inputs = Matrix::matrix_component_multiplication(&d_out_dnet, &errors).unwrap();
    //             let temp = &Matrix::matrix_multiplication(&d_net_w, &weight_changes_without_inputs.transpose()).unwrap();
    //             delta_weights = Matrix::matrix_addition(&delta_weights, &temp.transpose()).unwrap();
    //             delta_biases = Matrix::matrix_addition(&delta_biases, &Matrix::matrix_component_multiplication(&errors.get_single_col(i), &d_out_dnet).unwrap()).unwrap()
    //         }
    //         delta_weights = Matrix::scalar_multiplication(&delta_weights, learning_rate / expected.cols as f32);
    //
    //         delta_biases = Matrix::scalar_multiplication(&delta_biases, learning_rate / expected.cols as f32);
    //         if layer_index != 0 {
    //             // Calculate the error of the layer below
    //             errors = Matrix::matrix_multiplication(&self.layers[layer_index].weights.transpose(), &errors).unwrap();
    //         }
    //         self.layers.get_mut(layer_index).unwrap().weights = Matrix::matrix_subtraction(&self.layers.get(layer_index).unwrap().weights, &delta_weights).unwrap();
    //         self.layers.get_mut(layer_index).unwrap().change_bias(delta_biases);
    //     }
    // }
}

impl Debug for Network {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for layer in self.layers.iter() {
            write!(f, "Layer: \n{:?}\n", layer)?
        }
        Ok(())
    }
}

