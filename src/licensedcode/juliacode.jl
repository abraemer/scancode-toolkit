const MSet = Dict{Int,Int}

function build_set_and_tids_mset(token_ids)
    tids_mset = MSet()

    for tid in token_ids
        # this skips already matched token ids that are -1
        tid == -1 && continue
        tids_mset[tid] = get(()->0, tids_mset, tid) + 1
    end

    return BitSet(keys(tids_mset)), tids_mset
end

set_counter(set::BitSet) = length(set)
set_counter(set::MSet) = sum(values(set))

set_intersector(set1::BitSet, set2::BitSet) = intersect(set1, set2)
function set_intersector(set1::MSet, set2::MSet)
    length(set1) > length(set2) && ((set1,set2) = (set2,set1))
    ret = MSet()
    for (k, v) in set1
        !haskey(set2, k) && continue
        ret[k] = min(v, set2[k])
    end
    return ret
end

set_high_intersection_filter(set::BitSet, cutoff) = filter(<=(cutoff), set)
set_high_intersection_filter(set::MSet, cutoff) = filter(pair -> pair.first<=cutoff, set)

function compare_token_sets(qset, iset, len_legalese, min_matched_length_high, min_matched_length; minimum_containment=0, high_resemblance_threshold=0.8)
    intersection = set_intersector(qset, iset)
    length(intersection) == 0 && return nothing,nothing
    high_intersection = set_high_intersection_filter(intersection, len_legalese)
    length(high_intersection) == 0 && return nothing,nothing
    length(set_counter(high_intersection)) < min_matched_length_high && return nothing,nothing

    rule_length = set_counter(iset)
    matched_length = set_counter(intersection)
    matched_length < min_matched_length && return nothing, nothing

    union_len = set_counter(qset) + rule_length - matched_length
    resemblance = matched_length / union_len
    containment = matched_length / rule_length
    containment < minimum_containment && return nothing, nothing

    amplified_resemblance = resemblance^2
    score_vec1 = (;
        is_highly_resemblant=round(resemblance; digits=1) >= high_resemblance_threshold,
        containment=round(containment; digits=1),
        resemblance=round(amplified_resemblance; digits=1),
        matched_length=round(Int, matched_length / 20))

    score_vec2 = (;
        is_highly_resemblant=resemblance >= high_resemblance_threshold,
        containment=containment,
        resemblance=amplified_resemblance,
        matched_length=matched_length
    )

    return (score_vec1,score_vec2), high_intersection

end

const ScoreVector = @NamedTuple{is_highly_resemblant::Bool, containment::Float64, resemblance::Float64, matched_length::Int64}

struct RuleInfo
    min_matched_length_unique::Int
    min_matched_length::Int
    min_high_matched_length_unique::Int
    min_high_matched_length::Int
end
    
function convert_rule_list(rules_by_rid)
    return [RuleInfo(
        pyconvert(Any, r.get_min_matched_length(true)),
        pyconvert(Any, r.get_min_matched_length(false)),
        pyconvert(Any, r.get_min_high_matched_length(true)),
        pyconvert(Any, r.get_min_high_matched_length(false))) for r in rules_by_rid]
end

function compute_candidates(token_ids, len_legalese, rules_by_rid, sets_by_rid, msets_by_rid,
     matchable_rids, top=50, high_resemblance=false, high_resemblance_threshold=0.8)
    # collect query-side sets used for matching
    qset, qmset = build_set_and_tids_mset(token_ids)

    # @info "compute_candidates" typeof(token_ids) typeof(len_legalese) typeof(rules_by_rid) typeof(sets_by_rid) typeof(msets_by_rid) typeof(matchable_rids) typeof(top) typeof(high_resemblance) typeof(high_resemblance_threshold)
    # typeof(token_ids) = PyIterable{Any}
    # typeof(len_legalese) = Int64
    # typeof(rules_by_rid) = Vector{RuleInfo} (alias for Array{RuleInfo, 1})
    # typeof(sets_by_rid) = Vector{Union{Nothing, BitSet}} (alias for Array{Union{Nothing, BitSet}, 1})
    # typeof(msets_by_rid) = Vector{Union{Nothing, Dict{Int64, Int64}}} (alias for Array{Union{Nothing, Dict{Int64, Int64}}, 1})
    # typeof(matchable_rids) = PySet{Any}
    # typeof(top) = Int64
    # typeof(high_resemblance) = Bool
    # typeof(high_resemblance_threshold) = Float64


    # perform two steps of ranking:
    # step one with tid sets and step two with tid multisets for refinement

    ############################################################################
    # step 1 is on token id sets:
    ############################################################################

    sortable_candidates = Tuple{Tuple{ScoreVector,ScoreVector}, Int, RuleInfo, BitSet}[]

    for (rid, rule) in enumerate(rules_by_rid)
        rid -= 1 # julia python compat
        rid in matchable_rids || continue

        scores_vectors, high_set_intersection = compare_token_sets(qset, sets_by_rid[rid+1], len_legalese, rule.min_high_matched_length_unique, rule.min_matched_length_unique)

        if !isnothing(scores_vectors)
            svr, svf = scores_vectors
            if (!high_resemblance || (high_resemblance && svr.is_highly_resemblant && svf.is_highly_resemblant))
                # @info "" scores_vectors rid rule high_set_intersection
                push!(sortable_candidates, (scores_vectors, rid, rule, high_set_intersection))
            end
        end
    end

    length(sortable_candidates) == 0 && return sortable_candidates

    sort!(sortable_candidates; rev=true, by=x->x[1])

    ####################################################################
    # step 2 is on tids multisets
    ####################################################################
    # keep only the 10 x top candidates
    sortable_candidates_new = eltype(sortable_candidates)[]
    for (k , (_score_vectors, rid, rule, high_set_intersection)) in enumerate(sortable_candidates)
        k >= 10*top && break
        scores_vectors, _intersection = compare_token_sets(qmset, msets_by_rid[rid+1], len_legalese, rule.min_high_matched_length, rule.min_matched_length)

        if !isnothing(scores_vectors)
            svr, svf = scores_vectors
            if (!high_resemblance || (high_resemblance && svr.is_highly_resemblant && svf.is_highly_resemblant))
                push!(sortable_candidates_new, (scores_vectors, rid, rule, high_set_intersection))
            end
        end
    end

    length(sortable_candidates_new) == 0 && return sortable_candidates_new

    # rank candidates
    return sort!(sortable_candidates_new; rev=true)[1:min(top, length(sortable_candidates_new))]
end