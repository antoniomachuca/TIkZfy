from abc import ABC, abstractmethod

from core.models import ImageTensor, TikzTokens


class ImageToTikzUseCase(ABC):
    """
    Inbound port defining the mathematical contract for the core Use Case.

    This interface abstracts the orchestration of the image-to-markup pipeline,
    ensuring that outer controllers (Adapters, APIs, Scripts) can execute the
    domain logic without coupling to specific orchestrator implementations.
    """

    @abstractmethod
    def execute(self, image: ImageTensor) -> TikzTokens:
        """
        Executes the primary translation logic from spatial tensor representation
        to geometric markup.

        Args:
            image (ImageTensor): The statically validated, immutable tensor
                                 representation of the input image.

        Returns:
            TikzTokens: The strictly bounded and immutable sequence of generative
                        LaTeX markup.

        Raises:
            DomainError: If logical constraints are violated during the generative process.
        """
        pass
