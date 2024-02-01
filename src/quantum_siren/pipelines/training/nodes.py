"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.12
"""
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

import plotly.graph_objects as go

import mlflow
import logging

from .models import Model
from .optimizer import QNG, Adam

optimizers = {
    "QNG": QNG,
    "Adam": Adam,
}

log = logging.getLogger(__name__)


class Instructor:
    def __init__(
        self,
        n_layers,
        n_qubits,
        vqc_ansatz,
        iec_ansatz,
        data_reupload,
        learning_rate,
        shots,
        report_figure_every_n_steps,
        optimizer,
        output_interpretation,
        loss,
        seed,
    ) -> None:
        # this sets a global seed, that, according to documentation, affects the
        # weight initialization and dataloader
        torch.manual_seed(seed)

        self.steps_till_summary = report_figure_every_n_steps

        self.model = Model(
            n_qubits,
            shots,
            vqc_ansatz,
            iec_ansatz,
            n_layers,
            data_reupload,
            output_interpretation,
        )

        if optimizer == "QNG":
            self.optim = QNG(
                params=self.model.parameters(),
                lr=learning_rate,
                qnode=self.model.qnode,
                argnum=None,
            )
        elif optimizer == "Adam":
            self.optim = Adam(params=self.model.parameters(), lr=learning_rate)
        else:
            raise KeyError(f"No optimizer {optimizer} in {optimizers}")
        # self.optim = torch.optim.Adam(lr=learning_rate, params=self.model.parameters())

        self.metrics = {
            "mse": self.mse,
            "ssim": self.ssim,
            "fft_ssim": self.fft_ssim,
            "psnr": self.psnr,
        }

        # set the sign for the loss depending on the metric type
        # a "-1" means "maximize this metric" and a "1" means "minimize"
        if loss == "fft_ssim":
            self.loss = self.fft_ssim
            self.loss_sign = -1
        elif loss == "mse":
            self.loss = self.mse
            self.loss_sign = 1
        elif loss == "psnr":
            self.loss = self.psnr
            self.loss_sign = -1
        elif loss == "ssim":
            self.loss = self.ssim
            self.loss_sign = -1
        else:
            raise KeyError(f"No loss {loss} in {self.metrics}")

        # del self.metrics[loss]

    def cost(self, *args):
        return self.loss(*args) * self.loss_sign

    def fft_ssim(self, pred, target):
        if self.sidelength == -1:
            return -1
        pred_spectrum = torch.fft.fft2(pred.view(self.sidelength, self.sidelength))
        pred_spectrum = torch.fft.fftshift(pred_spectrum)

        target_spectrum = torch.fft.fft2(target.view(self.sidelength, self.sidelength))
        target_spectrum = torch.fft.fftshift(target_spectrum)

        val_abs = self.ssim(
            torch.log(pred_spectrum.abs()), torch.log(target_spectrum.abs())
        )
        val_phase = self.ssim(pred_spectrum.angle(), target_spectrum.angle())

        return (
            val_abs + val_phase
        ) / 2  # because we want to match phase and amplitude but keep the result <=1

    def ssim(self, pred, target):
        if self.sidelength == -1:
            return -1
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        val = ssim(
            pred.reshape(1, 1, self.sidelength, self.sidelength),
            target.reshape(1, 1, self.sidelength, self.sidelength),
        )

        return val

    def psnr(self, pred, target):
        if self.sidelength == -1:
            return -1
        psnr = PeakSignalNoiseRatio(data_range=1.0)
        val = psnr(
            pred.reshape(1, 1, self.sidelength, self.sidelength),
            target.reshape(1, 1, self.sidelength, self.sidelength),
        )

        return val

    def mse(self, pred, target):
        val = ((pred - target) ** 2).mean()

        return val

    def set_sidelength(self, dataloader):
        if len(dataloader.dataset.shape) == 3:
            self.sidelength = dataloader.dataset.sidelength
        elif len(dataloader.dataset.shape) == 2:
            self.sidelength = -1
        else:
            raise ValueError(f"Unsupported shape {dataloader.dataset.shape}")

    def train(self, dataloader, steps):
        self.set_sidelength(dataloader)

        for step in range(steps):
            metrics = {}
            loss_val = 0

            preds = []
            targets = []
            coords = []
            # Iterate the dataloader
            for coord, target in iter(dataloader):
                pred = self.model(coord)

                # # Idea is to create predictions, target and input independent
                # # of the actual data and batch size, which allows reshaping them later
                # preds = [*preds, *pred.flatten()]
                # targets = [*targets, *target.flatten()]
                # coords = [*coords, *coord.flatten()]

                loss = self.cost(pred, target)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                loss_val += loss.item()
            loss_val /= len(dataloader)

            coords = dataloader.dataset.coords
            preds = self.model(coords)
            targets = dataloader.dataset.values

            for name, metric in self.metrics.items():
                mlflow.log_metric(name, metric(preds, targets) / len(dataloader), step)

            if not step % self.steps_till_summary:
                # print(self.params)
                # print(f"Step {step}:\t Loss: {loss_val.item()}\t SSIM: {ssim_val}")

                if self.sidelength != -1:
                    fig = go.Figure(
                        data=go.Heatmap(
                            z=preds.cpu()
                            .view(self.sidelength, self.sidelength)
                            .detach()
                            .numpy(),
                            colorscale="RdBu",
                            zmid=0,
                        )
                    )
                    fig.update_layout(
                        yaxis=dict(scaleanchor="x", autorange="reversed"),
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                else:
                    fig = go.Figure(
                        data=[
                            go.Scatter(
                                x=coords,
                                y=preds.cpu().detach().numpy(),
                                mode="lines",
                                name="Prediction",
                            ),
                            go.Scatter(
                                x=coords, y=targets, mode="lines", name="Target"
                            ),
                        ]
                    )
                    fig.update_layout(
                        yaxis=dict(range=[-1.1, 1.1]),
                        plot_bgcolor="rgba(0,0,0,0)",
                    )

                mlflow.log_figure(fig, f"prediction_step_{step}.html")
                # print(f"Params: {params}")
                # img_grad = gradient(model_output, coords)
                # img_laplacian = laplace(model_output, coords)

                # fig, axes = plt.subplots(1,3, figsize=(18,6))
                # axes[1].imshow(img_grad.norm(dim=-1).cpu().view(sidelength,sidelength).detach().numpy())
                # axes[2].imshow(img_laplacian.cpu().view(sidelength,sidelength).detach().numpy())
                # plt.show()
                log.debug(f"Step {step}:\t Loss: {loss_val}")

        return self.model


def generate_instructor(
    n_layers,
    n_qubits,
    vqc_ansatz,
    iec_ansatz,
    data_reupload,
    learning_rate,
    shots,
    report_figure_every_n_steps,
):
    instructor = Instructor(
        n_layers,
        n_qubits,
        vqc_ansatz,
        iec_ansatz,
        data_reupload,
        learning_rate,
        shots,
        report_figure_every_n_steps,
    )

    return {"instructor": instructor}


def training(
    n_layers,
    n_qubits,
    vqc_ansatz,
    iec_ansatz,
    data_reupload,
    learning_rate,
    shots,
    report_figure_every_n_steps,
    optimizer,
    output_interpretation,
    loss,
    dataloader,
    steps,
    seed,
):
    instructor = Instructor(
        n_layers,
        n_qubits,
        vqc_ansatz,
        iec_ansatz,
        data_reupload,
        learning_rate,
        shots,
        report_figure_every_n_steps,
        optimizer,
        output_interpretation,
        loss,
        seed,
    )

    model = instructor.train(dataloader, steps)

    logging.info("Logging Model to MlFlow")
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="qameraman",
    )

    return {"model": model}
