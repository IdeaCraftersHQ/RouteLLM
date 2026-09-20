"""Abstract base class and marker decorator for prompt routers.

Kept free of module-level dependencies on `routers.py` so router
implementations in subpackages can subclass `Router` without forming an
import cycle with the module that registers them in `ROUTER_CLS`.
"""

import abc


def no_parallel(cls):
    """Mark router class as non-parallelizable.

    Some routers (e.g., those with randomness or state mutation) cannot be
    safely parallelized. This decorator marks them.

    Parameters
    ----------
    cls : type
        Router class to mark.

    Returns
    -------
    type
        The decorated class.
    """
    cls.NO_PARALLEL = True

    return cls


class Router(abc.ABC):
    """Abstract base class for prompt routers.

    Routers compute a confidence score (0-1) indicating the likelihood that
    a prompt should be routed to the strong model. A threshold is applied
    to make the routing decision: if score >= threshold, route to strong;
    otherwise route to weak.
    """
    NO_PARALLEL = False

    @abc.abstractmethod
    def calculate_strong_win_rate(self, prompt):
        """Calculate confidence score for routing to strong model.

        Parameters
        ----------
        prompt : str
            Input prompt to evaluate.

        Returns
        -------
        float
            Confidence score in [0, 1] representing estimated win rate of the
            strong model. Score >= threshold routes to strong; score < threshold
            routes to weak.
        """
        pass

    def route(self, prompt, threshold, routed_pair):
        """Route prompt to strong or weak model based on threshold.

        Parameters
        ----------
        prompt : str
            Input prompt to route.
        threshold : float
            Decision threshold in [0, 1]. If calculate_strong_win_rate >=
            threshold, route to strong; otherwise route to weak.
        routed_pair : ModelPair
            Pair of strong and weak model names.

        Returns
        -------
        str
            Name of model to route to (either routed_pair.strong or
            routed_pair.weak).
        """
        if self.calculate_strong_win_rate(prompt) >= threshold:
            return routed_pair.strong
        else:
            return routed_pair.weak

    def __str__(self):
        """Return router class name as string."""
        from routellm.routers.routers import NAME_TO_CLS
        return NAME_TO_CLS[self.__class__]
