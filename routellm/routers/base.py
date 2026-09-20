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

    def route_with_score(self, prompt, threshold, routed_pair):
        """Route a prompt and report the score the decision rested on.

        The scoring entry point for callers that need the win rate as
        well as the pick, such as the tier walk that records a decision
        path. `calculate_strong_win_rate` is called exactly once.

        A subclass that overrides `route` but not this method is taken
        as deciding by its own means: its `route` is called and the
        score is reported as None, rather than scoring the prompt a
        second time just to fill the field in.

        Parameters
        ----------
        prompt : str
            Input prompt to route.
        threshold : float
            Decision threshold in [0, 1]. If the win rate is greater
            than or equal to it, the strong model is chosen.
        routed_pair : ModelPair
            Pair of strong and weak model names.

        Returns
        -------
        tuple[str, float or None]
            The chosen model name, and the win rate behind it. None
            means this router reports no score.
        """
        if type(self).route is not Router.route:
            return self.route(prompt, threshold, routed_pair), None

        win_rate = self.calculate_strong_win_rate(prompt)
        model = routed_pair.strong if win_rate >= threshold else routed_pair.weak

        return model, win_rate

    def route(self, prompt, threshold, routed_pair):
        """Route prompt to strong or weak model based on threshold.

        Delegates to `route_with_score`, so the pick and the score can
        never disagree. A subclass overriding this method replaces the
        decision for both entry points.

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
        return self.route_with_score(prompt, threshold, routed_pair)[0]

    def __str__(self):
        """Return the name this router class is registered under."""
        from routellm.routers.registry import name_for

        return name_for(type(self))
