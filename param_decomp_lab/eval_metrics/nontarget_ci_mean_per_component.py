from typing import Literal

from param_decomp.base_config import BaseConfig
from param_decomp_lab.eval_metrics.ci_mean_per_component import _CIMeanPerComponentBase


class NontargetCIMeanPerComponentConfig(BaseConfig):
    type: Literal["NontargetCIMeanPerComponent"] = "NontargetCIMeanPerComponent"


class NontargetCIMeanPerComponent(_CIMeanPerComponentBase[NontargetCIMeanPerComponentConfig]):
    """`CIMeanPerComponent` fed by the nontarget eval loop (targeted runs).

    Identical accumulation; only the output-key prefix and routing differ.
    """

    short_name = "NtgtCIMeanPerComp"
    eval_distribution = "nontarget"
    _key_prefix = "nontarget_"
