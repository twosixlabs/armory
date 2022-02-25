from pydantic import BaseModel, validator


class Attack(BaseModel):
    name: str
    module: str
    knowledge: str
    kwargs: dict

    @validator("knowledge")
    def validate_knowledge(cls, v):
        if v not in ["white", "black"]:
            raise ValueError(
                f"invalid attack.knowledge: {v}...must be ['white'|'black']"
            )
        return v


class Dataset(BaseModel):
    name: str
    module: str
    framework: str
    batch_size: int


class Defense(BaseModel):
    name: str
    type: str
    module: str
    kwargs: dict


class Metric(BaseModel):
    means: bool
    perturbation: str
    record_metric_per_sample: bool
    task: list


class Model(BaseModel):
    name: str
    module: str
    weights_file: str = None
    wrapper_kwargs: dict
    model_kwargs: dict
    fit_kwargs: dict
    fit: bool


class Scenario(BaseModel):
    name: str
    module: str
    kwargs: dict


class Experiment(BaseModel):
    _description: str = None
    adhoc: bool = None  # TODO Figure out what this is for
    attack: Attack = None
    dataset: Dataset
    defense: Defense = None
    metric: Metric = None
    model: Model
    scenario: Scenario


if __name__ == "__main__":

    x = Experiment(
        name="seth",
        description="bob",
        adhoc=True,
        attack=Attack(name="a", module="a", knowledge="white"),
    )
    print(x)
    print(x.json())

    y = Experiment.parse_raw(x.json())
    print(f"y {y}")
