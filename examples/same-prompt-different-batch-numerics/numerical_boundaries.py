"""Float32 arithmetic and decision-boundary teaching examples; no device inference."""
import argparse
import json
import struct
from pathlib import Path

def f32(value):
    return struct.unpack('f',struct.pack('f',value))[0]

def add(a,b):
    return f32(f32(a)+f32(b))

def choose(scores):
    return max(range(len(scores)),key=lambda i:scores[i])

def demonstrate():
    a,b,c=100000000.0,-100000000.0,1.0
    left=add(add(a,b),c)
    right=add(a,add(b,c))
    assert (left,right)==(1.0,0.0)
    repeated_left=[add(add(a,b),c) for _ in range(100)]
    repeated_right=[add(a,add(b,c)) for _ in range(100)]
    assert set(repeated_left)=={1.0} and set(repeated_right)=={0.0}
    near=[f32(1.0000),f32(1.0001)]
    near_changed=[f32(1.0002),f32(1.0001)]
    far=[f32(1.0),f32(2.0)]
    far_changed=[f32(1.0002),f32(2.0)]
    assert choose(near)==1 and choose(near_changed)==0
    assert choose(far)==choose(far_changed)==1
    return {
        'scope':'CPU teaching arithmetic only; not evidence of a specific model or accelerator kernel defect.',
        'float32_grouping':{'left':left,'right':right,'exact_real_sum':1,'repeats_per_path':100,'each_path_repeatable':True},
        'near_boundary':{'before':near,'after':near_changed,'argmax_before':choose(near),'argmax_after':choose(near_changed)},
        'far_boundary':{'before':far,'after':far_changed,'argmax_before':choose(far),'argmax_after':choose(far_changed)},
        'error_bound':{'statement':'If every candidate error is bounded by epsilon and top1-top2 gap > 2*epsilon, the winner is preserved.','strict_inequality_required':True}
    }

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('numerical-results.json'))
    args=parser.parse_args()
    args.output.write_text(json.dumps(demonstrate(),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Verified float32 grouping, 100 repeats per path, and two argmax boundary examples.')
