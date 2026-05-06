use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use std::sync::OnceLock;

#[derive(Debug, Clone)]
struct Metrics {
    behavior_mse: f64,
    task_loss: f64,
    accuracy: f64,
    original_pred_match_rate: f64,
}

#[derive(Debug, Clone)]
struct ScoreRecord {
    component_id: String,
    component_index: usize,
    sample_row_count: usize,
    metrics: Metrics,
    slice_metrics: Vec<SliceRecord>,
}

#[derive(Debug, Clone)]
struct SliceRecord {
    name: String,
    count: usize,
    global_rows: Vec<i64>,
    metrics: Metrics,
}

static THREAD_POOL_SET: OnceLock<()> = OnceLock::new();

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn score_rank_one_linear_components(
    py: Python<'_>,
    inputs: PyReadonlyArray2<'_, f32>,
    labels: PyReadonlyArray1<'_, i64>,
    reference_logits: PyReadonlyArray2<'_, f32>,
    components_u: PyReadonlyArray2<'_, f32>,
    components_v: PyReadonlyArray2<'_, f32>,
    component_ids: Vec<String>,
    row_indices: PyReadonlyArray1<'_, i64>,
    slice_names: Vec<String>,
    slice_offsets: PyReadonlyArray1<'_, i64>,
    slice_indices: PyReadonlyArray1<'_, i64>,
    rust_threads: usize,
) -> PyResult<PyObject> {
    configure_rayon(rust_threads);

    let inputs = inputs.as_array();
    let labels = labels.as_array();
    let reference_logits = reference_logits.as_array();
    let components_u = components_u.as_array();
    let components_v = components_v.as_array();
    let row_indices = row_indices.as_array();
    let slice_offsets = slice_offsets.as_array();
    let slice_indices = slice_indices.as_array();

    validate_shapes(
        inputs.shape(),
        labels.shape(),
        reference_logits.shape(),
        components_u.shape(),
        components_v.shape(),
        component_ids.len(),
        row_indices.shape(),
        slice_names.len(),
        &slice_offsets.to_vec(),
        &slice_indices.to_vec(),
    )?;

    let n_rows = inputs.shape()[0];
    let out_dim = reference_logits.shape()[1];
    let in_dim = inputs.shape()[1];
    let inputs_slice = inputs.as_slice().expect("contiguous inputs");
    let labels_slice = labels.as_slice().expect("contiguous labels");
    let reference_slice = reference_logits
        .as_slice()
        .expect("contiguous reference logits");
    let u_slice = components_u.as_slice().expect("contiguous components_u");
    let v_slice = components_v.as_slice().expect("contiguous components_v");
    let row_indices_slice = row_indices.as_slice().expect("contiguous row indices");
    let slice_offsets_vec = slice_offsets.to_vec();
    let slice_indices_vec = slice_indices.to_vec();
    let ref_pred = argmax_rows(reference_slice, n_rows, out_dim);

    let records: Vec<ScoreRecord> = py.allow_threads(|| {
        component_ids
            .par_iter()
            .enumerate()
            .map(|(component_index, component_id)| {
                score_component(
                    component_index,
                    component_id,
                    inputs_slice,
                    labels_slice,
                    reference_slice,
                    u_slice,
                    v_slice,
                    &ref_pred,
                    row_indices_slice,
                    &slice_names,
                    &slice_offsets_vec,
                    &slice_indices_vec,
                    n_rows,
                    in_dim,
                    out_dim,
                )
            })
            .collect()
    });

    records_to_py(py, records)
}

fn configure_rayon(rust_threads: usize) {
    if rust_threads == 0 {
        return;
    }
    let _ = THREAD_POOL_SET.get_or_init(|| {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(rust_threads)
            .build_global();
    });
}

#[allow(clippy::too_many_arguments)]
fn validate_shapes(
    inputs_shape: &[usize],
    labels_shape: &[usize],
    reference_shape: &[usize],
    u_shape: &[usize],
    v_shape: &[usize],
    component_id_count: usize,
    row_indices_shape: &[usize],
    slice_name_count: usize,
    slice_offsets: &[i64],
    slice_indices: &[i64],
) -> PyResult<()> {
    if inputs_shape.len() != 2 {
        return Err(value_error("inputs must have shape [n_rows, in_dim]"));
    }
    if labels_shape != [inputs_shape[0]] {
        return Err(value_error("labels must have shape [n_rows]"));
    }
    if u_shape.len() != 2 || v_shape.len() != 2 {
        return Err(value_error("component factors must be rank-2 arrays"));
    }
    if reference_shape != [inputs_shape[0], u_shape[1]] {
        return Err(value_error(
            "reference_logits must have shape [n_rows, out_dim]",
        ));
    }
    if u_shape[0] != v_shape[0] || v_shape[1] != inputs_shape[1] {
        return Err(value_error(
            "component factors must have shapes [n_components, out_dim] and [n_components, in_dim]",
        ));
    }
    if component_id_count != u_shape[0] {
        return Err(value_error("component_ids length must equal n_components"));
    }
    if row_indices_shape != [inputs_shape[0]] {
        return Err(value_error("row_indices must have shape [n_rows]"));
    }
    if slice_name_count == 0 {
        if !slice_offsets.is_empty() || !slice_indices.is_empty() {
            return Err(value_error("slice arrays require slice_names"));
        }
    } else if slice_offsets.len() != slice_name_count + 1 {
        return Err(value_error(
            "slice_offsets length must be len(slice_names) + 1",
        ));
    } else if slice_offsets.first() != Some(&0)
        || slice_offsets.last() != Some(&(slice_indices.len() as i64))
    {
        return Err(value_error(
            "slice_offsets must start at 0 and end at len(slice_indices)",
        ));
    }
    for &index in slice_indices {
        if index < 0 || index as usize >= inputs_shape[0] {
            return Err(value_error("slice_indices must be local row indices"));
        }
    }
    Ok(())
}

fn value_error(message: &str) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(message.to_string())
}

#[allow(clippy::too_many_arguments)]
fn score_component(
    component_index: usize,
    component_id: &str,
    inputs: &[f32],
    labels: &[i64],
    reference_logits: &[f32],
    components_u: &[f32],
    components_v: &[f32],
    ref_pred: &[usize],
    row_indices: &[i64],
    slice_names: &[String],
    slice_offsets: &[i64],
    slice_indices: &[i64],
    n_rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> ScoreRecord {
    let ablated = ablated_logits_for_component(
        component_index,
        inputs,
        reference_logits,
        components_u,
        components_v,
        n_rows,
        in_dim,
        out_dim,
    );
    let metrics = metrics_for_rows(
        reference_logits,
        &ablated,
        labels,
        ref_pred,
        n_rows,
        out_dim,
        None,
    );
    let slice_metrics = slice_names
        .iter()
        .enumerate()
        .filter_map(|(slice_index, name)| {
            let start = slice_offsets[slice_index] as usize;
            let end = slice_offsets[slice_index + 1] as usize;
            let local_rows: Vec<usize> = slice_indices[start..end]
                .iter()
                .map(|&idx| idx as usize)
                .collect();
            if local_rows.is_empty() {
                return None;
            }
            let global_rows = local_rows
                .iter()
                .map(|&row| row_indices[row])
                .collect::<Vec<_>>();
            Some(SliceRecord {
                name: name.clone(),
                count: local_rows.len(),
                global_rows,
                metrics: metrics_for_rows(
                    reference_logits,
                    &ablated,
                    labels,
                    ref_pred,
                    n_rows,
                    out_dim,
                    Some(&local_rows),
                ),
            })
        })
        .collect();

    ScoreRecord {
        component_id: component_id.to_string(),
        component_index,
        sample_row_count: n_rows,
        metrics,
        slice_metrics,
    }
}

fn ablated_logits_for_component(
    component_index: usize,
    inputs: &[f32],
    reference_logits: &[f32],
    components_u: &[f32],
    components_v: &[f32],
    n_rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let u_offset = component_index * out_dim;
    let v_offset = component_index * in_dim;
    let mut output = vec![0.0_f32; n_rows * out_dim];
    output
        .par_chunks_mut(out_dim)
        .enumerate()
        .for_each(|(row, out_row)| {
            let input_row = &inputs[row * in_dim..(row + 1) * in_dim];
            let v = &components_v[v_offset..v_offset + in_dim];
            let scale = dot(input_row, v);
            for col in 0..out_dim {
                out_row[col] =
                    reference_logits[row * out_dim + col] - scale * components_u[u_offset + col];
            }
        });
    output
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

fn argmax_rows(values: &[f32], n_rows: usize, width: usize) -> Vec<usize> {
    (0..n_rows)
        .map(|row| {
            let row_values = &values[row * width..(row + 1) * width];
            argmax(row_values)
        })
        .collect()
}

fn argmax(values: &[f32]) -> usize {
    let mut best_index = 0;
    let mut best_value = values[0];
    for (index, &value) in values.iter().enumerate().skip(1) {
        if value > best_value {
            best_index = index;
            best_value = value;
        }
    }
    best_index
}

fn metrics_for_rows(
    reference_logits: &[f32],
    ablated_logits: &[f32],
    labels: &[i64],
    ref_pred: &[usize],
    n_rows: usize,
    out_dim: usize,
    rows: Option<&[usize]>,
) -> Metrics {
    let owned_rows;
    let rows = match rows {
        Some(rows) => rows,
        None => {
            owned_rows = (0..n_rows).collect::<Vec<_>>();
            &owned_rows
        }
    };
    let mut sq_sum = 0.0_f64;
    let mut ce_sum = 0.0_f64;
    let mut correct = 0_usize;
    let mut pred_match = 0_usize;
    for &row in rows {
        let start = row * out_dim;
        let end = start + out_dim;
        for col in 0..out_dim {
            let delta = ablated_logits[start + col] as f64 - reference_logits[start + col] as f64;
            sq_sum += delta * delta;
        }
        let pred = argmax(&ablated_logits[start..end]);
        let label = labels[row] as usize;
        if pred == label {
            correct += 1;
        }
        if pred == ref_pred[row] {
            pred_match += 1;
        }
        ce_sum += cross_entropy_row(&ablated_logits[start..end], label);
    }
    let count = rows.len().max(1) as f64;
    Metrics {
        behavior_mse: round_to(sq_sum / (count * out_dim as f64), 10),
        task_loss: round_to(ce_sum / count, 8),
        accuracy: round_to(correct as f64 / count, 8),
        original_pred_match_rate: round_to(pred_match as f64 / count, 8),
    }
}

fn cross_entropy_row(logits: &[f32], label: usize) -> f64 {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = logits
        .iter()
        .map(|&value| ((value - max_logit) as f64).exp())
        .sum();
    max_logit as f64 + exp_sum.ln() - logits[label] as f64
}

fn round_to(value: f64, places: i32) -> f64 {
    let scale = 10_f64.powi(places);
    (value * scale).round() / scale
}

fn records_to_py(py: Python<'_>, records: Vec<ScoreRecord>) -> PyResult<PyObject> {
    let output = PyList::empty(py);
    for record in records {
        let dict = PyDict::new(py);
        dict.set_item("component_id", record.component_id)?;
        dict.set_item("component_index", record.component_index)?;
        dict.set_item("sample_row_count", record.sample_row_count)?;
        set_metric_items(&dict, &record.metrics)?;
        let slice_dict = PyDict::new(py);
        for slice in record.slice_metrics {
            let item = PyDict::new(py);
            item.set_item("count", slice.count)?;
            item.set_item("global_rows", slice.global_rows)?;
            set_metric_items(&item, &slice.metrics)?;
            slice_dict.set_item(slice.name, item)?;
        }
        dict.set_item("slice_metrics", slice_dict)?;
        output.append(dict)?;
    }
    Ok(output.into())
}

fn set_metric_items(dict: &Bound<'_, PyDict>, metrics: &Metrics) -> PyResult<()> {
    dict.set_item("ablated_behavior_mse", metrics.behavior_mse)?;
    dict.set_item("ablated_task_loss", metrics.task_loss)?;
    dict.set_item("ablated_accuracy", metrics.accuracy)?;
    dict.set_item(
        "ablated_original_pred_match_rate",
        metrics.original_pred_match_rate,
    )?;
    Ok(())
}

#[pymodule]
fn param_decomp_accel(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(score_rank_one_linear_components, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_component_matches_manual_ablation() {
        let inputs = vec![1.0, 2.0, 0.5, -1.0];
        let labels = vec![0, 1];
        let reference = vec![2.0, 0.5, -0.25, 1.0];
        let u = vec![0.5, -1.0];
        let v = vec![2.0, 0.25];
        let ref_pred = argmax_rows(&reference, 2, 2);

        let record = score_component(
            0,
            "c0",
            &inputs,
            &labels,
            &reference,
            &u,
            &v,
            &ref_pred,
            &[10, 11],
            &[],
            &[],
            &[],
            2,
            2,
            2,
        );

        assert_eq!(record.component_id, "c0");
        assert_eq!(record.sample_row_count, 2);
        assert!(record.metrics.behavior_mse > 0.0);
        assert_eq!(record.slice_metrics.len(), 0);
    }
}
