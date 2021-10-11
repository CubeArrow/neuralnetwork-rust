use std::fs::File;
use std::io;
use std::io::Read;

use crate::matrix::Matrix;

struct ImageHeader {
    magic_number: u32,
    number_of_images: u32,
    number_of_rows: u32,
    number_of_cols: u32,
}

struct LabelHeader {
    magic_number: u32,
    number_of_items: u32,
}

fn read_f32(f: &mut File) -> io::Result<f32> {
    Ok(read_u8(f)? as f32)
}

fn read_u8(f: &mut File) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    f.read_exact(&mut buf)?;
    Ok(u8::from_be_bytes(buf))
}

fn read_u32(f: &mut File) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

pub fn get_labels(path: String, batch_size: usize) -> io::Result<Vec<Matrix>>{
    let mut f = File::open(path).expect("could not open the file.");

    let image_header = LabelHeader {
        magic_number: read_u32(&mut f)?,
        number_of_items: read_u32(&mut f)?,
    };
    let mut result = Vec::with_capacity(image_header.number_of_items as usize / batch_size);
    for i in 0..image_header.number_of_items / batch_size as u32 {
        let mut values = Vec::with_capacity(batch_size);
        for k in 0..batch_size{
            values.push(vec![0.0; 10]);
            values.last_mut().unwrap()[read_u8(&mut f)? as usize] = 1.0;
        }
        result.push(Matrix::from_values(values).expect("Test"));
    }

    Ok(result)
}

pub fn get_input_vec(path: String, batch_size: usize) -> io::Result<Vec<Matrix>> {
    let mut f = File::open(path).expect("could not open the file.");

    let image_header = ImageHeader {
        magic_number: read_u32(&mut f)?,
        number_of_images: read_u32(&mut f)?,
        number_of_rows: read_u32(&mut f)?,
        number_of_cols: read_u32(&mut f)?,
    };
    let mut result = Vec::with_capacity(image_header.number_of_images as usize / batch_size);
    for i in 0..image_header.number_of_images / batch_size as u32 {
        let image_size = (image_header.number_of_rows * image_header.number_of_cols) as usize;
        let mut values = Vec::with_capacity(image_size);
        for k in 0..batch_size{
            values.push(vec![]);
            for j in 0..image_size {
                values[k].push(read_f32(&mut f).expect("Test") / 255.0)
            }
        }

        result.push(Matrix::from_values(values).expect("Test"));
    }

    Ok(result)
}