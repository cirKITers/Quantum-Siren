"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import training, generate_instructor


def create_pipeline(**kwargs) -> Pipeline:
    # nd_generate_instructor = node(
    #     generate_instructor,
    #     inputs={
    #     },
    #     outputs={"instructor": "instructor"},
    # )

    nd_training = node(
        training,
        inputs={
            "n_layers": "params:n_layers",
            "n_qubits": "params:n_qubits",
            "vqc_ansatz": "params:vqc_ansatz",
            "iec_ansatz": "params:iec_ansatz",
            "data_reupload": "params:data_reupload",
            "learning_rate": "params:learning_rate",
            "shots": "params:shots",
            "report_figure_every_n_steps": "params:report_figure_every_n_steps",
            "optimizer": "params:optimizer",
            # "instructor": "instructor",
            "model_input": "coordinates",
            "ground_truth": "values",
            "steps": "params:steps",
        },
        outputs={"model": "model"},
    )
    return pipeline(
        [nd_training],
        inputs={"coordinates": "coordinates", "values": "values"},
        outputs={"model": "model"},
        namespace="training",
    )
