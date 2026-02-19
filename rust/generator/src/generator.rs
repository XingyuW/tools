use rand::rng;
use rand::prelude::SliceRandom;
use rand_distr::{Distribution, Normal};

/// Generates a vector statistically centered at a target value using the rand 0.10+ API.
pub fn generate_centered_array(
    target_val: f64,
    n_points: usize,
    majority_ratio: f64,
    fluctuation_intensity: f64,
) -> Vec<f64> {
    // Initialize the random number generator using the new API
    let mut rng = rng();
    
    let n_exact = (n_points as f64 * majority_ratio).floor() as usize;
    let n_noise = n_points - n_exact;

    let mut data = Vec::with_capacity(n_points);

    // Construct the majority component
    for _ in 0..n_exact {
        data.push(target_val);
    }

    // Construct the fluctuating component
    // Note: Ensure handling of potential unwrap failures in production code
    let normal = Normal::new(target_val, fluctuation_intensity).unwrap();
    
    for _ in 0..n_noise {
        data.push(normal.sample(&mut rng));
    }

    // Randomize the sequence using the IndexedRandom trait
    data.shuffle(&mut rng);

    data
}