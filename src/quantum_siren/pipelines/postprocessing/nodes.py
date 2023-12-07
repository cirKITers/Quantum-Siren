"""
This is a boilerplate pipeline 'postprocessing'
generated using Kedro 0.18.12
"""

from skimage.transform import rescale
import torch
import mlflow
import plotly.graph_objects as go
import math
import scipy
import numpy as np
import matplotlib.colors as colors
import plotly.express as px


def predict(model, coordinates):
    model_output = model(coordinates)

    return {"prediction": model_output.view(-1).detach()}


def upscaling(model, coordinates, factor):
    sidelength = math.sqrt(coordinates.shape[0])
    upscaled_sidelength = int(sidelength * factor)

    upscaled_coordinates = torch.zeros(size=(upscaled_sidelength**2, 2))
    it = 0
    for x in torch.linspace(coordinates.min(), coordinates.max(), upscaled_sidelength):
        for y in torch.linspace(
            coordinates.min(), coordinates.max(), upscaled_sidelength
        ):
            upscaled_coordinates[it] = torch.tensor([x, y])
            it += 1

    model_output = model(upscaled_coordinates)

    pred_upscaled_fig = go.Figure(
        data=go.Heatmap(
            z=model_output.cpu()
            .view(upscaled_sidelength, upscaled_sidelength)
            .detach()
            .numpy(),
            colorscale="RdBu",
            zmid=0,
        )
    )
    pred_upscaled_fig.update_layout(
        yaxis=dict(scaleanchor="x", autorange="reversed"), plot_bgcolor="rgba(0,0,0,0)"
    )

    # mlflow.log_figure(fig, f"{factor}x_upscaled_prediction.html")

    return {
        "pred_upscaled_fig": pred_upscaled_fig,
        "upscaled_image": model_output.detach(),
        "upscaled_coordinates": upscaled_coordinates,
    }


def pixelwise_difference(prediction, ground_truth):
    sidelength = int(math.sqrt(prediction.shape[0]))

    difference = prediction - ground_truth.view(sidelength**2)

    pixelwise_diff_fig = go.Figure(
        data=go.Heatmap(
            z=difference.cpu().view(sidelength, sidelength).detach().numpy(),
            colorscale="RdBu",
            zmid=0,
        )
    )

    pixelwise_diff_fig.update_layout(
        yaxis=dict(scaleanchor="x", autorange="reversed"), plot_bgcolor="rgba(0,0,0,0)"
    )

    # mlflow.log_figure(fig, f"pixelwise_differences.html")

    return {"pixelwise_diff_fig": pixelwise_diff_fig}


def plot_gradients(model, ground_truth, coordinates):
    sidelength = int(math.sqrt(coordinates.shape[0]))

    coordinates.requires_grad = True
    prediction = model(coordinates)

    # ----------------------------------------------------------------
    # Gradient Prediction

    pred_dc = torch.autograd.grad(
        outputs=prediction.sum(),
        inputs=coordinates,
        grad_outputs=None,
        create_graph=True,
    )[
        0
    ]  # same shape as coordinates

    pred_dc_img = grads2img(
        pred_dc[..., 0].view(sidelength, sidelength),
        pred_dc[..., 1].view(sidelength, sidelength),
        sidelength,
    )

    pred_gradient_fig = px.imshow(pred_dc_img)

    # pred_normed_dc = pred_dc.norm(dim=-1)

    # fig = go.Figure(data =
    #                 go.Heatmap(z = pred_normed_dc.cpu().view(sidelength, sidelength).detach().numpy(), colorscale='RdBu', zmid=0)
    #             )

    # fig.update_layout(
    #     yaxis=dict(
    #         scaleanchor='x',
    #         autorange='reversed'
    #     ),
    #     plot_bgcolor='rgba(0,0,0,0)'
    # )

    # mlflow.log_figure(fig, f"prediction_gradients.html")

    # ----------------------------------------------------------------
    # Laplacian Prediction

    pred_dcxdc = torch.autograd.grad(
        outputs=pred_dc[..., 0].sum(),
        inputs=coordinates,
        grad_outputs=None,
        retain_graph=True,
    )[0]
    pred_dcydc = torch.autograd.grad(
        outputs=pred_dc[..., 1].sum(),
        inputs=coordinates,
        grad_outputs=None,
        retain_graph=True,
    )[0]

    # select dcxdcx and dcydcy (we do not need dcxdcy and dcydcx)
    pred_dcdc = torch.stack((pred_dcxdc[..., 0], pred_dcydc[..., 1]), axis=-1)

    # calculate the sum so that it acutally becomes the laplace
    pred_laplace_dcdc = pred_dcdc.sum(dim=-1)

    pred_laplacian_fig = go.Figure(
        data=go.Heatmap(
            z=pred_laplace_dcdc.cpu().view(sidelength, sidelength).detach().numpy(),
            colorscale="RdBu",
            zmid=0,
        )
    )

    pred_laplacian_fig.update_layout(
        yaxis=dict(scaleanchor="x", autorange="reversed"), plot_bgcolor="rgba(0,0,0,0)"
    )

    # mlflow.log_figure(fig, f"prediction_laplacian.html")

    # ----------------------------------------------------------------
    # Gradient Ground Truth

    gt_image = ground_truth.reshape(sidelength, sidelength)

    gt_dcx = scipy.ndimage.sobel(gt_image, axis=0)
    gt_dcy = scipy.ndimage.sobel(gt_image, axis=1)

    gt_dc_img = grads2img(gt_dcx, gt_dcy, sidelength)

    gt_gradient_fig = px.imshow(gt_dc_img)

    # gt_dc = torch.stack((gt_dcx, gt_dcy), dim=-1).view(-1, 2)
    # gt_normed_dc = gt_dc.norm(dim=-1)

    # fig = go.Figure(data =
    #                 go.Heatmap(z = gt_normed_dc.cpu().view(sidelength, sidelength).detach().numpy(), colorscale='RdBu', zmid=0)
    #             )

    # fig.update_layout(
    #     yaxis=dict(
    #         scaleanchor='x',
    #         autorange='reversed'
    #     ),
    #     plot_bgcolor='rgba(0,0,0,0)'
    # )

    # mlflow.log_figure(fig, f"gradients.html")

    # ----------------------------------------------------------------
    # Laplacian Ground Truth

    gt_laplace_dcdc = scipy.ndimage.laplace(gt_image)

    gt_laplacian_fig = go.Figure(
        data=go.Heatmap(z=gt_laplace_dcdc, colorscale="RdBu", zmid=0)
    )

    gt_laplacian_fig.update_layout(
        yaxis=dict(scaleanchor="x", autorange="reversed"), plot_bgcolor="rgba(0,0,0,0)"
    )

    # mlflow.log_figure(fig, f"laplacian.html")

    return {
        "pred_gradient_fig": pred_gradient_fig,
        "pred_laplacian_fig": pred_laplacian_fig,
        "gt_gradient_fig": gt_gradient_fig,
        "gt_laplacian_fig": gt_laplacian_fig,
    }


def grads2img(grads_x, grads_y, sidelength):
    """
    Thankfully adapted from https://github.com/vsitzmann/siren/blob/master/dataio.py#L55
    """
    if type(grads_x) == torch.Tensor:
        grads_x = grads_x.detach().numpy()
    if type(grads_y) == torch.Tensor:
        grads_y = grads_y.detach().numpy()

    grads_a = np.arctan2(grads_y, grads_x)
    grads_m = np.hypot(grads_y, grads_x)
    grads_hsv = np.zeros((sidelength, sidelength, 3), dtype=np.float32)
    grads_hsv[:, :, 0] = (grads_a + np.pi) / (2 * np.pi)
    grads_hsv[:, :, 1] = 1.0

    nPerMin = np.percentile(grads_m, 5)
    nPerMax = np.percentile(grads_m, 95)
    grads_m = (grads_m - nPerMin) / (nPerMax - nPerMin)
    grads_m = np.clip(grads_m, 0, 1)

    grads_hsv[:, :, 2] = grads_m
    grads_rgb = colors.hsv_to_rgb(grads_hsv)

    return grads_rgb


def calculate_spectrum(values):
    sidelength = int(math.sqrt(values.shape[0]))

    spectrum = torch.fft.fft2(values.view(sidelength, sidelength))
    spectrum = torch.fft.fftshift(spectrum)

    spectrum_abs_fig = go.Figure(
        data=go.Heatmap(z=torch.log(spectrum.abs()).numpy(), colorscale="gray")
    )
    spectrum_abs_fig.update_layout(
        yaxis=dict(scaleanchor="x", autorange="reversed"), plot_bgcolor="rgba(0,0,0,0)"
    )

    # mlflow.log_figure(fig, f"spectrum_abs.html")

    spectrum_phase_fig = go.Figure(
        data=go.Heatmap(z=spectrum.angle().numpy(), colorscale="gray")
    )
    spectrum_phase_fig.update_layout(
        yaxis=dict(scaleanchor="x", autorange="reversed"), plot_bgcolor="rgba(0,0,0,0)"
    )

    # mlflow.log_figure(fig, f"spectrum_phase.html")

    return {
        "spectrum_abs_fig": spectrum_abs_fig,
        "spectrum_phase_fig": spectrum_phase_fig,
    }
