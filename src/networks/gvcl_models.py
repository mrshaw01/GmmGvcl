from networks.gvcl_model_classes import MultiHeadFiLMCNN


class ZenkeNetFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize, taskcla, tau):
            heads = [t[1] for t in taskcla]
            super().__init__(inputsize, [(32, 3), (32, 3), "pool", (64, 3), (64, 3), "pool"], [512], heads, film_type="point", tau=tau)


class SMNISTNetFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize, taskcla, tau):
            heads = [t[1] for t in taskcla]
            super().__init__(inputsize, [], [256, 256], heads, film_type="point", tau=tau)
