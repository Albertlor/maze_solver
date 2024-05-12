import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


class StatisticalAnalyzer:
    def __init__(self):
        self.sample_mean_list = []

    def collect_samples(self,sample):
        sample_mean = np.mean(np.array(sample))
        self.sample_mean_list.append(sample_mean)

    def analyze_confidence_level(self):
        data = np.array(self.sample_mean_list)
        mean = np.mean(data)
        std_dev = np.std(data)
        n = len(data)
        confidence_level = 0.95
        degrees_freedom = n - 1
        t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
        margin_error = t_critical * (std_dev / np.sqrt(n))
        confidence_interval = (mean - margin_error, mean + margin_error)
        self.plot_confidence_interval(mean,confidence_interval)
        return mean, std_dev, confidence_interval

    def plot_confidence_interval(self,mean,confidence_interval):
        plt.figure(figsize=(8, 4))
        plt.plot(mean, 'ko', markersize=10, label='Sample Mean')
        plt.hlines(y=mean, xmin=0, xmax=1, color='gray', lw=2)
        plt.errorbar(x=0, y=mean, yerr=[[mean - confidence_interval[0]], [confidence_interval[1] - mean]], fmt='o', color='red', label='95% Confidence Interval')
        plt.xlim(-1, 1)
        plt.xticks([])
        plt.title('Confidence Interval for Mean Steps')
        plt.legend()
        plt.show()

    def percentage_within_std_dev(self,data,num_std_dev=1):
        # Calculate mean and standard deviation
        mean = np.mean(data)
        std_dev = np.std(data)

        # Calculate the percentage of data within one standard deviation from the mean
        within_std_dev = ((data > (mean - num_std_dev*std_dev)) & (data < (mean + num_std_dev*std_dev))).mean() * 100

        # Plotting for visualization
        plt.hist(data, bins=30, alpha=0.75, color='blue')
        plt.axvline(mean, color='red', label='Mean')
        plt.axvline(mean + std_dev, color='green', linestyle='--', label='Mean ± 1 SD')
        plt.axvline(mean - std_dev, color='green', linestyle='--')
        plt.title('Sample Distribution of Solving Steps per Episode')
        plt.xlabel('Steps required to solve maze per episode')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

        return within_std_dev, mean, std_dev

    def samples_less_than_step_num(self,threshold,data):
        count = sum(1 for x in data if x < threshold)

        # Calculate the percentage of data points less than the threshold
        percentage = (count / len(data)) * 100 if data else 0  # Handles empty list case by returning 0%
        
        return percentage

    def plot_histogram(self,data):
        plt.hist(data, bins=30, alpha=0.75, color='blue')
        plt.title('Sample Distribution of Solving Steps per Episode')
        plt.xlabel('Steps required to solve maze per episode')
        plt.ylabel('Frequency')
        plt.show()