import pennylane as qml


class ansaetze:
    @staticmethod
    def nothing(params):
        pass

    @staticmethod
    def circuit_19(params):
        """
        Generates a single layer of circuit_19
        See https://arxiv.org/pdf/1905.10876.pdf

        Args:
            params (torch.tensor|np.ndarray): Parameters that are
            being utilized in the layer.
            Expects form to be [n_qubits, n_gates_per_layer],
            where n_gates_per_layer=3 in this case.
            If None, then the number of required params per layer is returned.
        """
        if params is None:
            return 3

        for q, q_params in enumerate(params):
            qml.RX(q_params[0], wires=q)
            qml.RZ(q_params[1], wires=q)

        for q, q_params in enumerate(params):
            qml.CRX(
                q_params[2],
                wires=[
                    params.shape[0] - q - 1,
                    (params.shape[0] - q) % params.shape[0],
                ],
            )

    @staticmethod
    def circuit_18(params):
        """
        Generates a single layer of circuit_19
        See https://arxiv.org/pdf/1905.10876.pdf

        Args:
            params (torch.tensor|np.ndarray): Parameters that are
            being utilized in the layer.
            Expects form to be [n_qubits, n_gates_per_layer],
            where n_gates_per_layer=3 in this case.
            If None, then the number of required params per layer is returned.
        """
        if params is None:
            return 3

        for q, q_params in enumerate(params):
            qml.RX(q_params[0], wires=q)
            qml.RZ(q_params[1], wires=q)

        for q, q_params in enumerate(params):
            qml.CRZ(
                q_params[2],
                wires=[
                    params.shape[0] - q - 1,
                    (params.shape[0] - q) % params.shape[0],
                ],
            )

    @staticmethod
    def default(params, **kwargs):
        for q, q_params in enumerate(params):
            qml.RX(q_params[0], wires=q)
            qml.RY(q_params[1], wires=q)

    @staticmethod
    def spread_layers(params, **kwargs):
        for q, q_params in enumerate(params):
            if 2 * q + 1 > params.shape[0] - 1:
                break
            qml.RY(q_params[0], wires=2 * q)
            qml.RY(q_params[1], wires=2 * q + 1)

    @staticmethod
    def spread_layers_limit(params, limit=2, **kwargs):
        for q, q_params in enumerate(params):
            if 2 * q >= limit:
                break

            qml.RY(q_params[0], wires=2 * q)
            qml.RY(q_params[1], wires=2 * q + 1)
