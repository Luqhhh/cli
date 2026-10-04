"""Pairing, inactive rows and supervision boundaries for the fixed V1 probe."""
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import probe_v1_768_supervision_transfer as probe


def supervision():
    return dict(weights=torch.tensor([.2,0.,.8,.5]),targets=torch.tensor([0,1,1,0]),
                labels=torch.tensor([0,1,0,0]),original_alpha=torch.zeros(4))


def test_population_keeps_additions_and_removals_under_one_order():
    a=supervision(); b=supervision(); b['weights']=torch.tensor([0.,.3,.4,0.])
    assert probe.common_population(a,b).tolist()==[0,1,2,3]


def test_weighted_loss_ignores_inactive_labels_and_matches_explicit_smoothed_ce():
    t=supervision(); logits=torch.tensor([[1.,-1.],[9.,-9.],[-.1,.3],[.5,.2]],requires_grad=True)
    actual=probe.weighted_loss(logits,t,torch.arange(4),2,.1)
    expected=(F.cross_entropy(logits,t['targets'],label_smoothing=.1,reduction='none')*t['weights']).sum()/t['weights'].sum()
    torch.testing.assert_close(actual,expected)
    actual.backward(); assert torch.count_nonzero(logits.grad[1])==0


def test_zero_mass_batch_rejected():
    with pytest.raises(ValueError,match='zero supervision'):
        probe.weighted_loss(torch.ones(1,2),supervision(),torch.tensor([1]),2,.1)


def test_identical_supervision_produces_bitwise_identical_heads_and_rng():
    torch.manual_seed(5); x=torch.randn(4,8)
    initial=probe.CosineHead(8,2).state_dict(); t=supervision()
    fit=dict(probe.HEAD_FIT,epochs=2,batch_size=2)
    heads,evidence=probe.fit_heads(x,(t,t),initial,torch.arange(4),fit,device='cpu')
    for key in initial:
        assert torch.equal(heads['control512'].state_dict()[key],heads['candidate768'].state_dict()[key])
    assert evidence['control512']==evidence['candidate768']


def test_changed_supervision_keeps_rng_and_indices_but_changes_parameters():
    torch.manual_seed(5); x=torch.randn(4,8); initial=probe.CosineHead(8,2).state_dict()
    a=supervision(); b=supervision(); b['targets']=1-b['targets']; b['weights']=torch.ones(4)
    heads,proof=probe.fit_heads(x,(a,b),initial,torch.arange(4),dict(probe.HEAD_FIT,epochs=2,batch_size=4),device='cpu')
    assert proof['control512']['sample_order_sha256']==proof['candidate768']['sample_order_sha256']
    assert proof['control512']['dropout_rng_sha256']==proof['candidate768']['dropout_rng_sha256']
    assert not torch.equal(heads['control512'].weight,heads['candidate768'].weight)


def test_target_rows_cannot_be_reordered_or_relabelled():
    t=supervision(); t.update(kept=t['weights']>0,pseudo=torch.zeros(4,dtype=torch.bool),agreement=torch.ones(4),
                            image_paths=['a','b','c','d'],class_names=['x','y'])
    rows=[dict(image_path=p,label=str(int(y))) for p,y in zip(t['image_paths'],t['labels'])]
    probe.targets_valid(t,rows,['x','y'])
    with pytest.raises(ValueError,match='row identity'):
        probe.targets_valid(t,list(reversed(rows)),['x','y'])
    t['labels']=1-t['labels']
    with pytest.raises(ValueError,match='original labels'):
        probe.targets_valid(t,rows,['x','y'])
