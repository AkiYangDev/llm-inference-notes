"""Read the original numeric NPZ exports without modifying the measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

DATA=Path(__file__).resolve().parents[2]/'assets/eplb-count-npz-reading'
EXPECTED={
    'A_protected':'5f7efc91576b713075d95402b7168ac4d887678c5e4b3ddc93644143ed0c2a73',
    'B_community':'eda480e8f19ac3af6094e5ebf7248c9de07c318dd106dc2d85adff934014b376',
}


def summarize():
    result={}
    mappings=[]
    for arm, expected_hash in EXPECTED.items():
        path=DATA/f'{arm}_global_counts.npz'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected_hash
        with np.load(path,allow_pickle=False) as data:
            counts=data['counts']
            roles=data['roles']
            mapping=data['physical_to_logical_map']
            assert counts.shape==(200,43,272) and mapping.shape==(43,272)
            assert counts.dtype==np.int64
            assert set(roles.tolist())<= {'target','draft'}
            assert np.all(counts>=0)
            targets=counts[roles=='target']
            assert np.all(targets.sum(axis=(1,2))==24768)
            assert np.all(targets[:,0,:].sum(axis=1)==576)
            assert np.all(counts[roles=='draft']==0)
            per_slot=targets[:,0,:].sum(axis=0)
            logical=np.zeros(int(mapping.max())+1,dtype=np.int64)
            np.add.at(logical,mapping[0],per_slot)
            assert logical.sum()==per_slot.sum()
            result[arm]=dict(
                sha256=expected_hash,shape=list(counts.shape),dtype=str(counts.dtype),
                ranks_already_summed=16,
                records_by_role={role:int((roles==role).sum()) for role in np.unique(roles)},
                global_total=int(counts.sum()),
                first_roles=roles[:12].tolist(),
                first_forward_ids=data['forward_pass_ids'][:12].tolist(),
                first_ring_slots=data['slots'][:12].tolist(),
                first_record_route_totals=counts.sum(axis=(1,2))[:12].tolist(),
                layer0_first16_target_sums=per_slot[:16].tolist(),
                layer0_first16_target_means=(per_slot[:16]/len(targets)).tolist(),
                layer0_logical_total=int(logical.sum()))
            mappings.append(mapping.copy())
    assert np.array_equal(*mappings)
    assert result['A_protected']['global_total']==4953600
    assert result['B_community']['global_total']==2476800
    return dict(source_date='2026-10-31',same_layout=True,
                scope='Independent runs; equal append counts, not token-aligned target forwards.',
                arms=result)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('count-summary.json'))
    args=parser.parse_args()
    args.output.write_text(json.dumps(summarize(),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Verified original NPZ hashes, axes, roles, route totals and logical aggregation.')
