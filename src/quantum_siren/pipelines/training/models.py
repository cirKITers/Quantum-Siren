import pennylane as qml

import torch

from .ansaetze import ansaetze

import logging

log = logging.getLogger(__name__)


class Model(torch.nn.Module):
    # class Module(torch.nn.Module):
    def __init__(
        self,
        n_qubits: int,
        shots: int,
        vqc_ansatz: str,
        iec_ansatz: str,
        n_layers: int,
        data_reupload: bool,
        output_interpretation: int,
        max_workers,
    ) -> None:
        super().__init__()

        log.info(f"Creating Model with {n_qubits} Qubits, {n_layers} Layers.")

        self.shots = None if shots == "None" else shots
        self.max_workers = None if max_workers == "None" else max_workers
        self.n_qubits = n_qubits
        # Following the ideas from https://doi.org/10.48550/arXiv.2008.08605
        # we add an additional layer to "sourround" our encoding
        self.n_layers = n_layers  # number of "visible" layers
        self._n_layers_p1 = n_layers + 1  # number of actual layers for weight init etc.

        self.iec = getattr(ansaetze, iec_ansatz, ansaetze.nothing)
        self.vqc = getattr(ansaetze, vqc_ansatz, ansaetze.nothing)

        if output_interpretation >= 0:
            output_interpretation = int(output_interpretation)
            assert output_interpretation < self.n_qubits, (
                f"Output interpretation parameter {output_interpretation} "
                "can either be a qubit (integer smaller n_qubits) or <0 (all qubits)"
            )

        self.output_interpretation = output_interpretation

        self.data_reupload = data_reupload

        dev = qml.device(
            "default.qubit",
            wires=self.n_qubits,
            shots=self.shots,
            max_workers=self.max_workers,
        )

        self.qnode = qml.QNode(self.circuit, dev, interface="torch")
        # print(qml.draw(self.circuit)(torch.rand(self._n_layers_p1, n_qubits, self.vqc(None)), torch.tensor([[0,1]])))
        self.qlayer = qml.qnn.TorchLayer(
            self.qnode,
            {"weights": [self._n_layers_p1, n_qubits, self.vqc(None)]},
        )

    def circuit(self, weights, inputs=None):
        if inputs is None:
            inputs = self._inputs
        else:
            self._inputs = inputs

        dru = torch.zeros(len(weights))
        if self.data_reupload != 0:
            dru[:: int(1 / self.data_reupload)] = 1

        # when iterating weights, the first dim. is the layer, the second is qubits
        for layer, layer_params in enumerate(weights[:-1]):  # N of (N+1) layers
            self.vqc(layer_params)
            qml.Barrier()
            if layer == 0 or dru[layer] == 1:
                self.iec(
                    torch.stack([inputs] * self.n_qubits),
                )  # half because the coordinates already have 2 dims

        self.vqc(weights[-1])  # the N+1 layer

        if self.output_interpretation < 0:
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        else:
            return qml.expval(qml.PauliZ(self.output_interpretation))

    def predict(self, context, model_input):
        if type(model_input) != torch.Tensor:
            model_input = torch.tensor(model_input)
        return self.forward(model_input)

    def forward(self, model_input):
        if self.output_interpretation < 0:
            out = torch.mean(self.qlayer(model_input), axis=1)
        else:
            out = self.qlayer(model_input)

        return out
