from typing import Protocol, runtime_checkable


@runtime_checkable
class SchedulerUncertaintyMixin(Protocol):
    
    timestep_after_step: int
    timestep_end_step: int

@runtime_checkable
class SchedulerUncertaintyClassConditionedMixin(Protocol):
    
    class_conditioned: bool
    timestep_after_step: int
    timestep_end_step: int

