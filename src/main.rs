use crate::matrix::Matrix;
use crate::network::{backpropagate, Network};
use crate::utils::cost;
use rand::seq::SliceRandom;

mod matrix;
mod layer;
mod network;
mod utils;

fn sigmoid(input: f32) -> f32 {
    input.exp() / (input.exp() + 1.0)
}

fn d_sigmoid(input: f32) -> f32 {
    input * (1.0 - input)
}

struct TrainValues {
    input: Matrix,
    desired_output: Matrix,
}

fn main() {
    let mut network = Network::new(&[2, 5, 5, 1]).unwrap();
    println!("{:?}", network);

    let mut train_values = vec![];

    let input = Matrix {
        values: vec![vec![1.0], vec![0.0]],
        rows: 2,
        cols: 1,
    };
    let desired = Matrix {
        values: vec![vec![1.0]],
        rows: 1,
        cols: 1,
    };
    train_values.push(TrainValues{input, desired_output:desired});

    let input = Matrix {
        values: vec![vec![0.0], vec![0.0]],
        rows: 2,
        cols: 1,
    };
    let desired_output = Matrix {
        values: vec![vec![0.0]],
        rows: 1,
        cols: 1,
    };
    train_values.push(TrainValues{input, desired_output});
    let input = Matrix {
        values: vec![vec![0.0], vec![1.0]],
        rows: 2,
        cols: 1,
    };
    let desired_output = Matrix {
        values: vec![vec![1.0]],
        rows: 1,
        cols: 1,
    };
    train_values.push(TrainValues{input, desired_output});

    let input = Matrix {
        values: vec![vec![1.0], vec![1.0]],
        rows: 2,
        cols: 1,
    };
    let desired_output = Matrix {
        values: vec![vec![0.0]],
        rows: 1,
        cols: 1,
    };
    train_values.push(TrainValues{input, desired_output});

    for _i in 0..50000 {
        let option = train_values.choose(&mut rand::thread_rng()).unwrap();
        let result = network.feedforward(option.input.clone(), &sigmoid).unwrap();
        network.backpropagate(&result, option.desired_output.clone(), &d_sigmoid, 0.1);
    }

    println!("{:?}", network);
    for option in train_values{
        let result = network.feedforward(option.input.clone(), &sigmoid).unwrap();
        println!("Input: {:?}, \nResult: {:?}, \nDesired: {:?}, \nCost: {:?}\n",option.input, result.last().unwrap().clone(), option.desired_output.clone(), cost(result.last().unwrap().clone(), option.desired_output.clone()));
    }
}
