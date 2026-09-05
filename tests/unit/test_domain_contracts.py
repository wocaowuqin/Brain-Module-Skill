from core.domain import CandidatePlan, CommitResult, RequestSpec, ResourceFootprint

def test_domain_contracts_are_serializable_values():
    req = RequestSpec(request_id=7, source=1, destinations=(2, 3), vnf_sequence=(0, 1))
    plan = CandidatePlan(request_id=req.request_id, accepted=True,
                         footprint=ResourceFootprint(cpu=1.0, nodes=(1,)))
    result = CommitResult(success=True, status="APPLIED", ledger_version="v1")
    assert plan.request_id == 7 and plan.footprint.cpu == 1.0
    assert result.success and result.ledger_version == "v1"
