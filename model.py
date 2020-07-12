from sklearn.pipeline import Pipeline


class JustTransformPipeline(Pipeline):
    def __init__(self):
        super().__init__()

    def just_transforms(self, X):
        """Applies all transforms to the data, without applying last 
        estimator.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step of
            the pipeline.
        """
        Xt = X
        for name, transform in self.steps[:-1]:
            Xt = transform.transform(Xt)
        return Xt
