use std::fs::File;
use std::io::Write;
use std::time::{Instant, SystemTime};

use chrono::{Local, Utc};
use rand::seq::IteratorRandom;

use crate::matrix::Matrix;
use crate::mnist_parser::{get_input_vec, get_labels};
use crate::network::Network;
use crate::utils::cost;

mod matrix;
mod layer;
mod network;
mod utils;
mod mnist_parser;

fn sigmoid(input: f32) -> f32 {
    input.exp() / (input.exp() + 1.0)
}

fn d_sigmoid(input: f32) -> f32 {
    input * (1.0 - input)
}

fn main() {
    let mut network = Network::new(&[784, 100, 100, 10]).unwrap();
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


    // let mut inputs = vec![];
    // let mut expected_results = vec![];
    // for i in 0..100{
    //     let mut temp_in: Vec<Vec<f32>> = vec![];
    //
    //     let mut temp_out: Vec<Vec<f32>> = vec![];
    //     for _j in 0..8{
    //         let x: f32 = rand::random::<f32>() * std::f32::consts::PI * 2.0;
    //         temp_in.push(vec![x]);
    //         temp_out.push((vec![(x.sin() + 1.0) / 2.0]));
    //     }
    //     inputs.push(Matrix::from_values(temp_in).unwrap());
    //     expected_results.push(Matrix::from_values(temp_out).unwrap());
    // }

    let epochs = 5000;
    for i in 0..epochs {
        let index = (0..inputs.len()).choose(&mut rand::thread_rng()).unwrap();
        let result = network.feedforward(inputs[index].clone(), &sigmoid).unwrap();

        network.backpropagate(&result, expected_results[index].clone(), &d_sigmoid, 0.001);
        if i % 500 == 0 || i == epochs - 1 {
            println!("Result: {:?}, \nDesired: {:?}, \nCost: {:?}\n", result.last().unwrap().clone(), expected_results[index].clone(), cost(&result.last().unwrap(), &expected_results[index]));
        }
    }
    let mut networkfile = File::create(format!("data/nn-{:?}.json", Local::now())).unwrap();
    write!(networkfile, "{}", serde_json::to_string(&network).unwrap()).unwrap();
}