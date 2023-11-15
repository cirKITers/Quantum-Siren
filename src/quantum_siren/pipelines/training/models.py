import pennylane as qml

import torch

from .ansaetze import ansaetze

import logging

log = logging.getLogger(__name__)


class Model(torch.nn.Module):
    # class Module(torch.nn.Module):
    def __init__(
        self,
        n_qubits,
        shots,
        vqc_ansatz,
        iec_ansatz,
        n_layers,
        data_reupload,
        output_interpretation,
    ) -> None:
        super().__init__()

        log.info(f"Creating Model with {n_qubits} Qubits, {n_layers} Layers.")

        self.shots = None if shots == "None" else shots
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.iec = getattr(ansaetze, iec_ansatz, ansaetze.nothing)
        self.vqc = getattr(ansaetze, vqc_ansatz, ansaetze.nothing)

        assert (
            output_interpretation == "all" or type(output_interpretation) == int
        ), f"Output interpretation parameter can either be a qubit (integer) or 'all'"
        self.output_interpretation = output_interpretation

        self.data_reupload = data_reupload

        dev = qml.device("default.qubit", wires=self.n_qubits, shots=self.shots)

        self.qnode = qml.QNode(self.circuit, dev, interface="torch")
        self.qlayer = qml.qnn.TorchLayer(
            self.qnode,
            {"weights": [n_layers, n_qubits, self.vqc(None)]},
        )

        self.initialize_params(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            n_gates_per_layer=self.vqc(None),
        )

    def initialize_params(self, n_qubits, n_layers, n_gates_per_layer):
        self.params = torch.nn.Parameter(
            torch.rand(size=(n_layers, n_qubits, n_gates_per_layer), requires_grad=True)
        )

    def circuit(self, weights, inputs=None):
        if inputs is None:
            inputs = self._inputs
        else:
            self._inputs = inputs

        dru = torch.zeros(len(weights))
        if self.data_reupload != 0:
            dru[:: int(1 / self.data_reupload)] = 1

        for l, l_params in enumerate(weights):
            if l == 0 or dru[l] == 1:
                self.iec(
                    torch.stack([inputs] * self.n_qubits),
                )  # half because the coordinates already have 2 dims

            self.vqc(l_params)

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def predict(self, context, model_input):
        if type(model_input) != torch.Tensor:
            model_input = torch.tensor(model_input)
        return self.forward(model_input)

    def forward(self, model_input):
        # return self.qlayer(model_input)

        if model_input.ndim == 2:
            out = torch.zeros(
                size=[
                    model_input.shape[0],
                ]
            )

            # with Pool(processes=4) as pool:
            #     out = pool.starmap(self.qnode, [[params, coord] for coord in model_input])

            for i, coord in enumerate(model_input):
                if self.output_interpretation == "all":
                    out[i] = torch.mean(self.qlayer(coord), axis=0)
                else:
                    out[i] = self.qlayer(coord)[self.output_interpretation]
        else:
            out = self.qlayer(model_input)[-1]

        return out
