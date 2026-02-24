#![allow(unsafe_op_in_unsafe_fn)]

mod generator;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
fn generate_centered_array(
    target_val: f64,
    n_points: usize,
    majority_ratio: f64,
    fluctuation_intensity: f64,
) -> PyResult<Vec<f64>> {
    if !(0.0..=1.0).contains(&majority_ratio) {
        return Err(PyValueError::new_err(
            "majority_ratio must be between 0.0 and 1.0",
        ));
    }
    if fluctuation_intensity <= 0.0 {
        return Err(PyValueError::new_err(
            "fluctuation_intensity must be greater than 0.0",
        ));
    }

    Ok(generator::generate_centered_array(
        target_val,
        n_points,
        majority_ratio,
        fluctuation_intensity,
    ))
}

#[pyfunction]
fn generate_time_axis(n_points: usize, sampling_rate: f64) -> PyResult<Vec<f64>> {
    if sampling_rate <= 0.0 {
        return Err(PyValueError::new_err(
            "sampling_rate must be greater than 0.0",
        ));
    }

    let dt = 1.0 / sampling_rate;
    let axis = (0..n_points).map(|i| i as f64 * dt).collect();
    Ok(axis)
}

#[pyfunction]
fn generate_signal_with_time(
    target_val: f64,
    n_points: usize,
    majority_ratio: f64,
    fluctuation_intensity: f64,
    sampling_rate: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let signal = generate_centered_array(
        target_val,
        n_points,
        majority_ratio,
        fluctuation_intensity,
    )?;
    let time = generate_time_axis(n_points, sampling_rate)?;
    Ok((time, signal))
}

#[pymodule]
fn signal_generator(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(generate_centered_array, module)?)?;
    module.add_function(wrap_pyfunction!(generate_time_axis, module)?)?;
    module.add_function(wrap_pyfunction!(generate_signal_with_time, module)?)?;
    Ok(())
}