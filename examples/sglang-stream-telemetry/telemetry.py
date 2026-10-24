"""Bounded n=1 response metadata retention; no transport or clock access.

An educational variant of the historical observer, not a production drop-in
for every SGLang/AISBench version. Counter validation happens separately.
"""
import copy
import json

SPEC_FIELDS = (
    'spec_accept_rate', 'spec_accept_length', 'spec_cap_length',
    'spec_block_accept_length', 'spec_num_correct_drafts',
    'spec_num_proposed_drafts', 'spec_verify_ct',
    'spec_correct_drafts_histogram', 'spec_cap_lens_histogram',
)


def retain_metadata(chunk, output):
    extra = output.extra_details_data
    usage = chunk.get('usage')
    if isinstance(usage, dict):
        prior = extra.get('prefix_cache_usage', {})
        p, c = usage.get('prompt_tokens'), usage.get('completion_tokens')
        details = usage.get('prompt_tokens_details')
        cached = details.get('cached_tokens') if isinstance(details, dict) else None
        value = {'usage_events': prior.get('usage_events', 0) + 1}
        if type(p) is not int or type(c) is not int or p < 0 or c < 0:
            value['status'] = 'INVALID_NUMERIC_USAGE'
        else:
            value.update(prompt_tokens=p, completion_tokens=c)
            if type(cached) is int and 0 <= cached <= p:
                value.update(status='OBSERVED', cached_tokens=cached)
            else:
                value.update(status='CACHED_TOKENS_NOT_OBSERVED', cached_tokens='NOT_OBSERVED')
        # usage is a response snapshot, not an increment to sum over chunks.
        extra['prefix_cache_usage'] = value

    extension = chunk.get('sglext')
    if not isinstance(extension, dict) or extension.get('spec_tokens_details') is None:
        return
    observer = extra.setdefault('spec_counter_observer', {'metadata_events': 0})
    observer['metadata_events'] += 1
    if observer['metadata_events'] != 1:
        extra['spec_capture_status'] = 'MULTIPLE_FINAL_METADATA'
        return
    details = extension['spec_tokens_details']
    if not isinstance(details, dict):
        extra['spec_capture_status'] = 'UNSUPPORTED_NON_N1_DETAILS'
        return
    retained = {key: copy.deepcopy(details[key]) for key in SPEC_FIELDS if key in details}
    try:
        encoded = json.dumps(retained, allow_nan=False).encode('utf-8')
    except (ValueError, TypeError):
        extra['spec_capture_status'] = 'INVALID_RESPONSE_METADATA'
        return
    if len(encoded) > 8192:
        extra['spec_capture_status'] = 'RESPONSE_METADATA_TOO_LARGE'
        return
    extra['sglang_spec_tokens_details'] = retained
    extra['spec_capture_status'] = 'RETAINED_UNVALIDATED'


def summarize_counts(extra, *, proposal_denominator_verified=False):
    """Only a source-bound actual proposal denominator may yield strict AR.

    A field named proposed can still mean a verify budget in another mode.
    The boolean is a provenance precondition, never an inference from values.
    """
    result = {'AL_output': 'NOT_OBSERVED', 'strict_AR': 'NOT_OBSERVED'}
    if extra.get('spec_capture_status') != 'RETAINED_UNVALIDATED':
        return result
    details = extra['sglang_spec_tokens_details']
    v = details.get('spec_verify_ct')
    a = details.get('spec_num_correct_drafts')
    p = details.get('spec_num_proposed_drafts')
    if v is None or v == 0:
        return result
    if type(v) is not int or v < 0:
        raise ValueError('Invalid counter values')
    for value in (a, p):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError('Invalid counter values')
    if a is not None and p is not None and a > p:
        raise ValueError('Accepted count exceeds returned proposed count')
    hist = details.get('spec_correct_drafts_histogram')
    if hist is not None:
        if (not isinstance(hist, list) or any(type(n) is not int or n < 0 for n in hist)
                or sum(hist) != v or (a is not None and sum(i*n for i, n in enumerate(hist)) != a)):
            raise ValueError('Histogram does not agree with response counts')
    usage = extra.get('prefix_cache_usage', {})
    if type(usage.get('completion_tokens')) is int:
        result['AL_output'] = usage['completion_tokens'] / v
    if proposal_denominator_verified and a is not None and p is not None and p > 0:
        result['strict_AR'] = a / p
    return result
