"""
This is a boilerplate pipeline 'postprocessing'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import (
    upscaling,
    pixelwise_difference,
    plot_gradients,
    calculate_spectrum,
    predict,
)


def create_pipeline(**kwargs) -> Pipeline:
    nd_predict = node(
        predict,
        inputs={
            "model": "model",
            "coordinates": "coordinates",
        },
        outputs={"prediction": "prediction"},
    )

    nd_upscaling = node(
        upscaling,
        inputs={
            "model": "model",
            "coordinates": "coordinates",
            "factor": "params:upscale_factor",
        },
        outputs={
            "pred_upscaled_fig": "pred_upscaled_fig",
            "upscaled_image": "upscaled_image",
            "upscaled_coordinates": "upscaled_coordinates",
        },
    )

    nd_pixelwise_diff = node(
        pixelwise_difference,
        inputs={"prediction": "prediction", "ground_truth": "ground_truth"},
        outputs={"pixelwise_diff_fig": "pixelwise_diff_fig"},
    )

    nd_plot_gradients = node(
        plot_gradients,
        inputs={
            "model": "model",
            "ground_truth": "ground_truth",
            "coordinates": "coordinates",
        },
        outputs={
            "pred_gradient_fig": "pred_gradient_fig",
            "pred_laplacian_fig": "pred_laplacian_fig",
            "gt_gradient_fig": "gt_gradient_fig",
            "gt_laplacian_fig": "gt_laplacian_fig",
        },
    )

    return pipeline(
        [
            nd_predict,
            nd_upscaling,
            nd_pixelwise_diff,
            nd_plot_gradients,
            node(
                calculate_spectrum,
                inputs={"values": "ground_truth"},
                outputs={
                    "spectrum_abs_fig": "gt_spectrum_abs_fig",
                    "spectrum_phase_fig": "gt_spectrum_phase_fig",
                },
            ),
            node(
                calculate_spectrum,
                inputs={"values": "prediction"},
                outputs={
                    "spectrum_abs_fig": "pred_spectrum_abs_fig",
                    "spectrum_phase_fig": "pred_spectrum_phase_fig",
                },
            ),
        ],
        inputs={
            "coordinates": "coordinates",
            "model": "model",
            "ground_truth": "values",
        },
        outputs={"upscaled_image": "upscaled_image"},
        namespace="postprocessing",
    )
