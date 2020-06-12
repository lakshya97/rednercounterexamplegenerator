from verifai.server import Server

# A fake server that bypasses the networking stuff.
class FakeServer(Server):
    def __init__(self, sampling_data, monitor, client_task, options={}):
        self.monitor = monitor
        self.lastValue = None
        self.task = client_task

        if sampling_data.sampler is not None:
            self.sampler_type = ('random' if sampling_data.sampler_type is None
                                 else sampling_data.sampler_type)
            self.sampler = sampling_data.sampler
            self.sample_space = (self.sampler.space
                                 if sampling_data.sample_space is None
                                 else sampling_data.sample_space)

        elif sampling_data.sampler_type is None:
            feature_space = {}
            for space_name in sampling_data.sample_space:
                space = sampling_data.sample_space[space_name]
                feature_space[space_name] = Feature(space)
            self.sample_space = FeatureSpace(feature_space)
            self.sampler_type = 'random'
            self.sampler = FeatureSampler.samplerFor(self.sample_space)
            self.sample_space = self.sampler.space

        else:
            feature_space = {}
            for space_name in sampling_data.sample_space:
                space = sampling_data.sample_space[space_name]
                feature_space[space_name] = Feature(space)
            self.sample_space = FeatureSpace(feature_space)
            params = (None if 'sampler_params' not in sampling_data
                      else sampling_data.sampler_params)
            self.sampler_type, self.sampler = choose_sampler(
                sample_space=self.sample_space,
                sampler_type=sampling_data.sampler_type,
                sampler_params=params
            )

        print("Initialized sampler")

    def evaluate_sample(self, sample):
        return self.monitor.evaluate(self.task.simulate(sample))

    def terminate(self):
        pass
