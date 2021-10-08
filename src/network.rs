use crate::layer::Layer;
use std::fmt::{Formatter, Debug};
use crate::matrix::Matrix;

#[derive(Clone)]
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
    pub fn feedforward(&self, input: Matrix, activaction_function: &dyn Fn(f32) -> f32) -> Result<Vec<Matrix>, String>{
        let mut res = vec![input];
        for i in 0..self.layers.len(){
            res.push(self.get_result_index(i, res.last().unwrap(), activaction_function)?);
        }
        return Ok(res);
    }
    pub fn get_result_index(&self, index: usize, input: &Matrix, activation_function: &dyn Fn(f32) -> f32) -> Result<Matrix, String>{
        return self.layers[index].clone().get_result(input, activation_function);
    }


    pub fn backpropagate(&mut self, result: &Vec<Matrix>, expected: Matrix, derivative_activation_function: &dyn Fn(f32) -> f32, learning_rate: f32){
        // Calculate the initial error
        let mut errors = Matrix::matrix_subtraction(&result.last().unwrap(), &expected).unwrap();
        for layer_index in (0..network.layers.len()).rev() {
            // Apply the derivative of the activation function to the output of the current layer
            let d_out_dnet = result[layer_index + 1].apply_function(derivative_activation_function);

            // get the inputs of the current layer
            let d_net_w = &result[layer_index];

            // Calculate the weights to be subtracted
            let weight_changes_without_inputs = Matrix::matrix_component_multiplication(&d_out_dnet, &errors).unwrap();
            let subtractable_weights = Matrix::scalar_multiplication(Matrix::matrix_multiplication(&d_net_w, &weight_changes_without_inputs.transpose()).unwrap(), learning_rate);

            network.layers.get_mut(layer_index).unwrap().weights = Matrix::matrix_subtraction(&network.layers.get(layer_index).unwrap().weights, &subtractable_weights.transpose()).unwrap();
            network.layers.get_mut(layer_index).unwrap().change_bias(Matrix::scalar_multiplication(Matrix::matrix_component_multiplication(&errors, &d_out_dnet).unwrap(), learning_rate));
            if layer_index != 0 {
                // Calculate the error of the layer below
                errors = Matrix::matrix_multiplication(&network.layers[layer_index].weights.transpose(), &errors).unwrap();
            }
        }
    }
}
impl Debug for Network{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for layer in self.layers.iter(){
            write!(f, "Layer: \n{:?}\n", layer)?
        }
        Ok(())
    }
}

