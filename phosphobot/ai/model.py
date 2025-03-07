class Model:
    """
    Base class for robotic models in the phosphobot framework, analogous to PyTorch's nn.Module.
    Subclasses must implement the forward method to define how inputs are processed.
    """

    def forward(self, *args, **kwargs):
        """
        Defines the forward pass of the model. Subclasses must override this method.

        Args:
            *args: Variable positional arguments (e.g., sensor inputs like camera frames).
            **kwargs: Variable keyword arguments (e.g., additional configuration).

        Returns:
            The model's output (e.g., predictions, actions).

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement the forward method")

    def __call__(self, *args, **kwargs):
        """
        Makes the model instance callable, delegating to the forward method.

        Args:
            *args: Variable positional arguments passed to forward.
            **kwargs: Variable keyword arguments passed to forward.

        Returns:
            The output of the forward method.
        """
        return self.forward(*args, **kwargs)
