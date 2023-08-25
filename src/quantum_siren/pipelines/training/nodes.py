"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.12
"""
from multiprocessing import Pool

import pennylane as qml
from pennylane import numpy as np

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

import math

import plotly.graph_objects as go

from .ansaetze import ansaetze

import mlflow


class Model(mlflow.pyfunc.PythonModel):
    def __init__(self, n_qubits, shots, vqc_ansatz, iec_ansatz, n_layers, data_reupload) -> None:
        self.shots = None if shots=="None" else shots
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.iec = getattr(ansaetze, iec_ansatz, ansaetze.nothing)
        self.vqc = getattr(ansaetze, vqc_ansatz, ansaetze.nothing)
        
        self.data_reupload = data_reupload

        dev = qml.device("default.qubit", wires=self.n_qubits, shots=self.shots)
        self.predict = qml.QNode(self.circuit, dev, interface="torch")

        self.params = self.initialize_params(n_qubits=self.n_qubits, n_layers=self.n_layers, n_gates_per_layer=self.vqc(None))

    def initialize_params(self, n_qubits, n_layers, n_gates_per_layer):
        return torch.rand(size=(n_layers,n_qubits,n_gates_per_layer), requires_grad=True)

    def circuit(self, model_input):
        for l, l_params in enumerate(self.params):
            if l == 0 or (l > 0 and self.data_reupload):
                self.iec(torch.stack([model_input]*(self.n_qubits//2)), limit=self.n_qubits-(l//2)) # half because the coordinates already have 2 dims

            self.vqc(l_params)

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]


class Instructor():
    def __init__(self, n_layers, n_qubits, vqc_ansatz, iec_ansatz, data_reupload, learning_rate, shots) -> None:


        self.steps_till_summary = 10
        
        self.model = Model(n_qubits, shots, vqc_ansatz, iec_ansatz, n_layers, data_reupload)
        self.optim = torch.optim.Adam(lr=learning_rate, params=[self.model.params])
    

    def cost(self, model_input):
        out = torch.zeros(size=[model_input.shape[0],])

        # with Pool(processes=4) as pool:
        #     out = pool.starmap(self.qnode, [[params, coord] for coord in model_input])
        
        for i, coord in enumerate(model_input):
            # out[i] = torch.mean(torch.stack(circuit(params, coord)), axis=0)
            out[i] = self.model.predict(coord)[-1]

        return out

    def ssim(self, pred, target):
        sidelength = int(math.sqrt(target.shape[1]))

        ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        val = ssim(pred.reshape(1, 1, sidelength, sidelength), target.reshape(1, 1, sidelength, sidelength))

        return val

    def mse(self, pred, target):
        val = ((pred - target)**2).mean()

        return val

    def train(self, model_input, ground_truth, steps, report_figure_every_n_steps):
        sidelength = int(math.sqrt(ground_truth.shape[1]))

        for step in range(steps):

            model_output = self.cost(model_input[0])
            model_output = model_output.reshape((1, model_output.shape[0], 1))

            loss_val = self.mse(model_output, ground_truth)
            ssim_val = self.ssim(model_output, ground_truth)

            mlflow.log_metric("Loss", loss_val.item(), step)
            mlflow.log_metric("SSIM", ssim_val, step)
            if not step % self.steps_till_summary:
                # print(self.params)
                # print(f"Step {step}:\t Loss: {loss_val.item()}\t SSIM: {ssim_val}")
                fig = go.Figure(data =
                    go.Heatmap(z = model_output.cpu().view(sidelength, sidelength).detach().numpy())
                )
                fig.update_layout(
                    yaxis=dict(
                        scaleanchor='x',
                        autorange='reversed'
                    ),
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                mlflow.log_figure(fig, f"prediction_step_{step}.html")
                # print(f"Params: {params}")
                # img_grad = gradient(model_output, coords)
                # img_laplacian = laplace(model_output, coords)

                # fig, axes = plt.subplots(1,3, figsize=(18,6))
                # axes[1].imshow(img_grad.norm(dim=-1).cpu().view(sidelength,sidelength).detach().numpy())
                # axes[2].imshow(img_laplacian.cpu().view(sidelength,sidelength).detach().numpy())
                # plt.show()

            self.optim.zero_grad()
            loss_val.backward()
            self.optim.step()

        return self.model


def generate_instructor(n_layers, n_qubits, vqc_ansatz, iec_ansatz, data_reupload, learning_rate, shots):
    instructor = Instructor(n_layers, n_qubits, vqc_ansatz, iec_ansatz, data_reupload, learning_rate, shots)

    return {
        "instructor": instructor
    }

def plot_ground_truth(ground_truth):
    sidelength = int(math.sqrt(ground_truth.shape[1]))
    fig = go.Figure(data =
        go.Heatmap(z = ground_truth.view(sidelength, sidelength).detach().numpy())
    )
    fig.update_layout(
        yaxis=dict(
            scaleanchor='x',
            autorange='reversed'
        ),
        plot_bgcolor='rgba(0,0,0,0)'
    )

    mlflow.log_figure(fig, f"ground_truth.html")

    return {
    }

def training(instructor, model_input, ground_truth, steps, report_figure_every_n_steps):
    model = instructor.train(model_input, ground_truth, steps, report_figure_every_n_steps)

    mlflow.pyfunc.log_model(python_model=model, artifact_path="qameraman", input_example=model_input.numpy()[0][0])

    return {
        "model": model
    }