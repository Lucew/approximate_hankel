import numpy as np


class ChangeSimulator:

    def __init__(self, length: int, change_point_position: int, random_generator: np.random.RandomState,
                 random_changepoint = True):

        # save the necessary information into the object
        self.length = length
        self.initial_changepoint_position = change_point_position
        self.random_generator = random_generator
        self.random_changepoint = random_changepoint

        # check that the change point comes before the signal end
        assert 0 < self.initial_changepoint_position < self.length, "The changepoint needs to be in the signal."

    def check_signal(self, signal: np.ndarray):
        assert len(signal) == self.length, f"Signal length is not correct. Expected: {self.length} got {len(signal)}."
        return signal

    def make_random_changepoint(self):
        changepoint_position = self.initial_changepoint_position
        if self.random_changepoint:
            start = int(self.initial_changepoint_position*0.5)
            end = self.initial_changepoint_position+int((self.length-self.initial_changepoint_position)*0.5)
            changepoint_position = self.random_generator.randint(start, end)
        return changepoint_position

    def exponential_decline(self, decline_length: int = None):

        # update the random change point
        changepoint_position = self.make_random_changepoint()

        # create and automatic decline length by using 20% of the signal lengths
        if decline_length is None:
            length = self.random_generator.randint(5,50)/100
            decline_length = int((self.length-changepoint_position)*length)

        # check that the decline length is shorter than the signal length
        assert decline_length < self.length, "The decline length needs to be shorter than the overall length."

        steady_before = np.ones(changepoint_position)
        exp_signal = np.exp(-np.linspace(0, 5, decline_length))
        steady_after = np.exp(-5) * np.ones(self.length-len(steady_before)-len(exp_signal))
        signal = np.concatenate((steady_before, exp_signal, steady_after))

        # add some noise
        signal += self.random_generator.normal(scale=0.001, size=len(signal))

        # make a signal length assertion
        assert len(signal) == self.length, "Length is not enough."

        return self.check_signal(signal)

    def frequency_change(self, frequency_before: int = None, frequency_after: int = None):

        # check whether we specified something (both have to be specified)
        if (frequency_before is None) ^ (frequency_after is None):
            raise ValueError("You can only specify both or none of the frequencies.")

        # check whether both have not been specified make some automated decisions
        if frequency_before is None and frequency_after is None:
            frequency_before = self.random_generator.randint(1, 10)
            if frequency_before > 5:
                frequency_after = self.random_generator.randint(1, 5)
            else:
                frequency_after = self.random_generator.randint(6, 10)

        # update the random change point
        changepoint_position = self.make_random_changepoint()

        # create a linear array before the change point
        t1 = np.linspace(0, 2*np.pi*frequency_before, changepoint_position)

        # create the array after the change point
        t2 = np.linspace(0, 2*np.pi*frequency_after, self.length-changepoint_position)

        # concatenate the signals
        t = np.concatenate((t1, t2))

        # compute the signal
        signal = np.sin(t)

        # add some noise
        signal += self.random_generator.normal(scale=0.001, size=len(signal))

        return self.check_signal(signal)

    def mean_change(self, mean_before: float = None, mean_after: float = None):

        # check whether we specified something (both have to be specified)
        if (mean_before is None) ^ (mean_after is None):
            raise ValueError("You can only specify both or none of the means.")

        # make default definition
        if mean_before is None and mean_after is None:
            mean_before = self.random_generator.randint(0, 100)/100
            if mean_before > 50:
                mean_after = self.random_generator.randint(0, 50)/100
            else:
                mean_after = self.random_generator.randint(51, 100)/100

        # check that the mean is between 0 and 1
        assert 0 <= mean_before <= 1 and 0 <= mean_after <= 1, "Specified means need to be between 0 and 1."

        # update the random change point
        changepoint_position = self.make_random_changepoint()

        # create the signal
        t1 = np.ones(changepoint_position)*mean_before
        t2 = np.ones(self.length-changepoint_position)*mean_after
        signal = np.concatenate((t1, t2))

        # add some noise
        signal += self.random_generator.normal(scale=0.001, size=len(signal))

        return self.check_signal(signal)

    def variance_change(self, variance_before: float = None, variance_after: float = None):

        # check whether we specified something (both have to be specified)
        if (variance_before is None) ^ (variance_after is None):
            raise ValueError("You can only specify both or none of the means.")

        # make default definition
        if variance_before is None and variance_after is None:
            variance_before = self.random_generator.randint(1, 100)/100
            if variance_before > 50:
                variance_after = self.random_generator.randint(1, 50)/100
            else:
                variance_after = self.random_generator.randint(60, 100)/100

        # update the random change point
        changepoint_position = self.make_random_changepoint()

        # create the signals
        t1 = self.random_generator.normal(scale=variance_before, size=changepoint_position)
        t2 = self.random_generator.normal(scale=variance_after, size=self.length-changepoint_position)
        signal = np.concatenate((t1, t2))

        return self.check_signal(signal)

    def yield_signals(self):

        # go over all the signals and create them
        generator_handles = [self.mean_change, self.variance_change, self.frequency_change, self.exponential_decline]
        for signal_generator in generator_handles:
            yield signal_generator.__name__, signal_generator()


def main():
    import matplotlib.pyplot as plt

    # make a signal generator
    gn = ChangeSimulator(1000, 500, np.random.RandomState(10))

    # simulate some signals
    for _ in range(2):
        for name, signal in gn.yield_signals():
            plt.plot(signal)
    plt.show()


if __name__ == "__main__":
    main()
