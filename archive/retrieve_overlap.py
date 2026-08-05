from consensuscnv.callsets.calls import Call


def retrieve_overlap(call_a: Call, call_b: Call) -> tuple[bool, float, int]:
    """Pairwise reference implementation"""
    reciprocal_overlap = 0.0
    distance = 0

    if call_a.chrom != call_b.chrom:
        return False, 0, 0

    overlap_start = max(call_a.start, call_b.start)
    overlap_end = min(call_a.end, call_b.end)

    if overlap_start < overlap_end:
        overlap_length = overlap_end - overlap_start
        reciprocal_overlap = overlap_length / max(
            call_a.end - call_a.start, call_b.end - call_b.start
        )
        distance = 0
    else:
        distance = min(abs(call_a.start - call_b.end), abs(call_b.start - call_a.end))

    return True, reciprocal_overlap, distance
