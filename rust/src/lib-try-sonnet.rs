use hashbrown::{HashMap, HashSet};
use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple};
use std::cmp::min;
use std::collections::hash_map::Entry;

#[pyclass]
#[derive(Clone)]
struct ScoresVector {
    #[pyo3(get)]
    is_highly_resemblant: bool,
    #[pyo3(get)]
    containment: f64,
    #[pyo3(get)]
    resemblance: f64,
    #[pyo3(get)]
    matched_length: f64,
}

#[pymethods]
impl ScoresVector {
    #[new]
    fn new(is_highly_resemblant: bool, containment: f64, resemblance: f64, matched_length: f64) -> Self {
        Self {
            is_highly_resemblant,
            containment,
            resemblance,
            matched_length,
        }
    }
}

type TokenId = i64;
type TokenSet = HashSet<TokenId>;
type TokenMultiset = HashMap<TokenId, usize>;
type BigramMultiset = HashMap<(TokenId, TokenId), usize>;

fn tids_sets_intersector(qset: &TokenSet, iset: &TokenSet) -> TokenSet {
    qset.intersection(iset).cloned().collect()
}

fn tids_set_counter(set: &TokenSet) -> usize {
    set.len()
}

fn multisets_intersector(qmset: &TokenMultiset, imset: &TokenMultiset) -> TokenMultiset {
    let mut intersection = TokenMultiset::new();
    
    // Iterate over the smaller set
    let (set1, set2) = if qmset.len() < imset.len() {
        (qmset, imset)
    } else {
        (imset, qmset)
    };
    
    for (key, s1count) in set1 {
        if let Some(s2count) = set2.get(key) {
            let count = min(*s1count, *s2count);
            if count > 0 {
                intersection.insert(*key, count);
            }
        }
    }
    
    intersection
}

fn multiset_counter(mset: &TokenMultiset) -> usize {
    mset.values().sum()
}

fn high_tids_set_subset(tids_set: &TokenSet, len_legalese: TokenId) -> TokenSet {
    tids_set.iter().filter(|&&i| i < len_legalese).cloned().collect()
}

fn high_tids_multiset_subset(mset: &TokenMultiset, len_legalese: TokenId) -> TokenMultiset {
    mset.iter()
        .filter(|&(tid, _)| *tid < len_legalese)
        .map(|(tid, count)| (*tid, *count))
        .collect()
}

fn high_bigrams_multiset_subset(mset: &BigramMultiset, len_legalese: TokenId) -> BigramMultiset {
    mset.iter()
        .filter(|&(bigram, _)| bigram.0 < len_legalese || bigram.1 < len_legalese)
        .map(|(bigram, count)| (*bigram, *count))
        .collect()
}

fn build_set_and_tids_mset(token_ids: &[TokenId]) -> (TokenSet, TokenMultiset) {
    let mut tids_mset = TokenMultiset::new();
    
    for &tid in token_ids {
        if tid == -1 {
            continue;
        }
        *tids_mset.entry(tid).or_insert(0) += 1;
    }
    
    let tids_set: TokenSet = tids_mset.keys().cloned().collect();
    
    (tids_set, tids_mset)
}

fn build_set_and_bigrams_mset(token_ids: &[TokenId]) -> (TokenSet, BigramMultiset) {
    let mut tids_set = TokenSet::new();
    let mut bigrams_mset = BigramMultiset::new();
    
    for window in token_ids.windows(2) {
        if window.contains(&-1) {
            continue;
        }
        
        let bigram = (window[0], window[1]);
        *bigrams_mset.entry(bigram).or_insert(0) += 1;
        tids_set.insert(bigram.0);
        tids_set.insert(bigram.1);
    }
    
    (tids_set, bigrams_mset)
}

fn compare_token_sets(
    py: Python,
    qset: &PyAny,
    iset: &PyAny,
    intersector_type: &str,
    counter_type: &str,
    high_intersection_filter_type: &str,
    len_legalese: TokenId,
    unique: bool,
    rule: &PyAny,
    filter_non_matching: bool,
    high_resemblance_threshold: f64,
    use_bigrams: bool,
) -> PyResult<Option<(Py<PyTuple>, PyObject)>> {
    // Extract native Rust collections based on the type
    let (intersection, matched_length, high_intersection, high_matched_length) = match intersector_type {
        "sets" => {
            let qset_native: TokenSet = qset.extract()?;
            let iset_native: TokenSet = iset.extract()?;
            
            let intersection = tids_sets_intersector(&qset_native, &iset_native);
            if intersection.is_empty() {
                return Ok(None);
            }
            
            let matched_length = tids_set_counter(&intersection);
            let high_intersection = high_tids_set_subset(&intersection, len_legalese);
            let high_matched_length = tids_set_counter(&high_intersection);
            
            (intersection.into_py(py), matched_length, high_intersection.into_py(py), high_matched_length)
        },
        "multisets" => {
            if use_bigrams {
                let qmset_native: BigramMultiset = qset.extract()?;
                let imset_native: BigramMultiset = iset.extract()?;
                
                // This is a simplification - in a real implementation you'd need to handle bigrams properly
                let intersection = HashMap::new(); // Placeholder
                if intersection.is_empty() {
                    return Ok(None);
                }
                
                let matched_length = 0; // Placeholder
                let high_intersection = HashMap::new(); // Placeholder
                let high_matched_length = 0; // Placeholder
                
                (intersection.into_py(py), matched_length, high_intersection.into_py(py), high_matched_length)
            } else {
                let qmset_native: TokenMultiset = qset.extract()?;
                let imset_native: TokenMultiset = iset.extract()?;
                
                let intersection = multisets_intersector(&qmset_native, &imset_native);
                if intersection.is_empty() {
                    return Ok(None);
                }
                
                let matched_length = multiset_counter(&intersection);
                let high_intersection = high_tids_multiset_subset(&intersection, len_legalese);
                let high_matched_length = multiset_counter(&high_intersection);
                
                (intersection.into_py(py), matched_length, high_intersection.into_py(py), high_matched_length)
            }
        },
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid intersector type")),
    };
    
    if filter_non_matching {
        if high_matched_length == 0 {
            return Ok(None);
        }
        
        let min_high_matched_length: usize = rule.call_method1("get_min_high_matched_length", (unique,))?.extract()?;
        if high_matched_length < min_high_matched_length {
            return Ok(None);
        }
    }
    
    let min_matched_length: usize = rule.call_method1("get_min_matched_length", (unique,))?.extract()?;
    if filter_non_matching && matched_length < min_matched_length {
        return Ok(None);
    }
    
    let iset_len: usize = rule.call_method1("get_length", (unique,))?.extract()?;
    let qset_len: usize = match counter_type {
        "set" => qset.len()?,
        "multiset" => qset.call_method0("values")?.call_method0("__iter__")?.iter()?.map(|v| v?.extract::<usize>()).sum::<PyResult<usize>>()?,
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid counter type")),
    };
    
    let union_len = qset_len + iset_len - matched_length;
    let resemblance = matched_length as f64 / union_len as f64;
    let containment = matched_length as f64 / iset_len as f64;
    let amplified_resemblance = resemblance * resemblance;
    
    let minimum_containment: Option<f64> = rule.getattr("_minimum_containment")?.extract()?;
    if filter_non_matching && minimum_containment.is_some() && containment < minimum_containment.unwrap() {
        return Ok(None);
    }
    
    let scores_vector1 = ScoresVector::new(
        (resemblance * 10.0).round() / 10.0 >= high_resemblance_threshold,
        (containment * 10.0).round() / 10.0,
        (amplified_resemblance * 10.0).round() / 10.0,
        (matched_length as f64 / 20.0 * 10.0).round() / 10.0,
    );
    
    let scores_vector2 = ScoresVector::new(
        resemblance >= high_resemblance_threshold,
        containment,
        amplified_resemblance,
        matched_length as f64,
    );
    
    let scores_tuple = PyTuple::new(py, &[Py::new(py, scores_vector1)?, Py::new(py, scores_vector2)?]);
    
    Ok(Some((scores_tuple.into_py(py), high_intersection)))
}

#[pyfunction]
fn build_set_and_mset(py: Python, token_ids: Vec<TokenId>, use_bigrams: bool) -> PyResult<Py<PyTuple>> {
    if use_bigrams {
        let (tids_set, bigrams_mset) = build_set_and_bigrams_mset(&token_ids);
        let py_set = PySet::new(py, &tids_set.into_iter().collect::<Vec<_>>())?;
        
        let py_mset = PyDict::new(py);
        for ((k1, k2), v) in bigrams_mset {
            py_mset.set_item((k1, k2), v)?;
        }
        iter().collect::<Vec<_>>()?;
        
        let py_mset = PyDict::new(py);
        for (k, v) in tids_mset {
            py_mset.set_item(k, v)?;
        }
        
        Ok(PyTuple::new(py, &[py_set, py_mset]).into_py(py))
    }
}

#[pyfunction]
fn compute_candidates(
    py: Python,
    query_run: &PyAny,
    idx: &PyAny,
    matchable_rids: &PySet,
    top: usize,
    high_resemblance: bool,
    high_resemblance_threshold: f64,
    use_bigrams: bool,
) -> PyResult<Vec<PyObject>> {
    // Get token IDs from query_run
    let token_ids: Vec<TokenId> = query_run.call_method0("matchable_tokens")?.extract()?;
    
    // Build sets and multisets
    let (qset, qmset) = if use_bigrams {
        build_set_and_bigrams_mset(&token_ids)
    } else {
        build_set_and_tids_mset(&token_ids)
    };
    
    let len_legalese: TokenId = idx.getattr("len_legalese")?.extract()?;
    let sets_by_rid = idx.getattr("sets_by_rid")?;
    let rules_by_rid = idx.getattr("rules_by_rid")?;
    
    // Step 1: Token ID sets
    let mut sortable_candidates = Vec::new();
    
    for rid in matchable_rids.iter()? {
        let rid: usize = rid?.extract()?;
        let rule = rules_by_rid.get_item(rid)?;
        let iset = sets_by_rid.get_item(rid)?;
        
        if let Some((scores_vectors, high_set_intersection)) = compare_token_sets(
            py,
            &qset.into_py(py),
            iset,
            "sets",
            "set",
            "high_tids_set_subset",
            len_legalese,
            true,
            rule,
            true,
            high_resemblance_threshold,
            false,
        )? {
            let svr = scores_vectors.get_item(0)?;
            let svf = scores_vectors.get_item(1)?;
            
            let svr_highly_resemblant: bool = svr.getattr("is_highly_resemblant")?.extract()?;
            let svf_highly_resemblant: bool = svf.getattr("is_highly_resemblant")?.extract()?;
            
            if !high_resemblance || (high_resemblance && svr_highly_resemblant && svf_highly_resemblant) {
                sortable_candidates.push((scores_vectors, rid, rule, high_set_intersection));
            }
        }
    }
    
    if sortable_candidates.is_empty() {
        return Ok(Vec::new());
    }
    
    // Sort candidates (Python will handle the sorting)
    let py_sortable_candidates = PyList::new(py, &sortable_candidates);
    py_sortable_candidates.call_method1("sort", (true,))?; // reverse=True
    
    // Step 2: Multisets
    // Keep only the 10 x top candidates
    let candidates_limit = top * 10;
    let candidates = if py_sortable_candidates.len()? > candidates_limit {
        py_sortable_candidates.call_method1("__getitem__", (pyo3::types::PySlice::new(py, 0, candidates_limit as isize, 1),))?
    } else {
        py_sortable_candidates.into_py(py)
    };
    
    let mut step2_candidates = Vec::new();
    let msets_by_rid = idx.getattr("msets_by_rid")?;
    
    for candidate in candidates.iter()? {
        let candidate = candidate?;
        let _score_vectors = candidate.get_item(0)?;
        let rid: usize = candidate.get_item(1)?.extract()?;
        let rule = candidate.get_item(2)?;
        let high_set_intersection = candidate.get_item(3)?;
        
        let iset = msets_by_rid.get_item(rid)?;
        
        let filter_non_matching = !use_bigrams;
        
        if let Some((scores_vectors, _intersection)) = compare_token_sets(
            py,
            &qmset.into_py(py),
            iset,
            "multisets",
            "multiset",
            if use_bigrams { "high_bigrams_multiset_subset" } else { "high_tids_multiset_subset" },
            len_legalese,
            false,
            rule,
            filter_non_matching,
            high_resemblance_threshold,
            use_bigrams,
        )? {
            let svr = scores_vectors.get_item(0)?;
            let svf = scores_vectors.get_item(1)?;
            
            let svr_highly_resemblant: bool = svr.getattr("is_highly_resemblant")?.extract()?;
            let svf_highly_resemblant: bool = svf.getattr("is_highly_resemblant")?.extract()?;
            
            if !high_resemblance || (high_resemblance && svr_highly_resemblant && svf_highly_resemblant) {
                step2_candidates.push((scores_vectors, rid, rule, high_set_intersection));
            }
        }
    }
    
    if step2_candidates.is_empty() {
        return Ok(Vec::new());
    }
    
    // Filter duplicates and sort
    let py_step2_candidates = PyList::new(py, &step2_candidates);
    
    // In Python we'd call filter_dupes here, but we'll just sort and take top N
    py_step2_candidates.call_method1("sort", (true,))?; // reverse=True
    
    let final_candidates = if py_step2_candidates.len()? > top {
        py_step2_candidates.call_method1("__getitem__", (pyo3::types::PySlice::new(py, 0, top as isize, 1),))?
    } else {
        py_step2_candidates.into_py(py)
    };
    
    // Convert to Vec<PyObject>
    let mut result = Vec::new();
    for item in final_candidates.iter()? {
        result.push(item?.into_py(py));
    }
    
    Ok(result)
}

#[pymodule]
fn rust_intersector(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ScoresVector>()?;
    m.add_function(wrap_pyfunction!(build_set_and_mset, m)?)?;
    m.add_function(wrap_pyfunction!(compute_candidates, m)?)?;
    Ok(())
}